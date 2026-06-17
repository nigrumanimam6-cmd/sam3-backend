# ============================================================================
#  SAM3 BACKEND · main.py
#  FastAPI + WebSocket. Dos capacidades sobre el modelo de IMAGEN de SAM 3:
#   1) Segmentar una imagen por texto        -> overlay cian   (type "image"/"text")
#   2) Procesar un VIDEO frame-por-frame     -> "capas" rastreadas (type "video")
#
#  El video se procesa cuadro a cuadro con el modelo de imagen + un rastreador
#  casero (IoU + centroide + apariencia). Devuelve, por objeto estable: color,
#  etiqueta, rastro y miniatura; y por frame: un overlay compuesto.
#
#  Protocolo "video":
#    Cliente -> {"type":"video","data":"<mp4 base64>","prompt":"person",
#                "min_presence":0.5, "max_frames":120, "stride":1}
#    Servidor-> {"type":"progress","done":i,"total":N}          (repetido)
#    Servidor-> {"type":"video_result","n_frames":N,"width":W,"height":H,
#                "objects":[{id,color,label,trail,thumb_png}],
#                "frames":[{frame, overlay_png}]}
# ============================================================================

import io
import os
import json
import base64
import asyncio
import tempfile
from collections import defaultdict

import numpy as np
import torch
import cv2
from PIL import Image
from scipy.optimize import linear_sum_assignment
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- Estado global: el modelo de imagen (compartido) ------------------------
_processor = None


def load_model():
    """Carga SAM 3 (modelo de imagen). Llamar UNA vez al arrancar."""
    global _processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    model = build_sam3_image_model()
    _processor = Sam3Processor(model)
    return _processor


# ============================================================================
#  1) SEGMENTACION DE IMAGEN (overlay cian) -- flujo de imagen
# ============================================================================
def segmentar(image, prompt):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = _processor.set_image(image)
        out = _processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = out["masks"], out["scores"]
    n = int(masks.shape[0])
    if n == 0:
        return None, 0, 0.0
    m = masks[0, 0].float().cpu().numpy()
    thr = 0.5 if m.max() <= 1.0 + 1e-3 else 0.0
    m_bin = m > thr
    H, W = m_bin.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[m_bin] = [0, 200, 255, 130]
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii"), n, float(scores[0].float().cpu())


# ============================================================================
#  2) MOTOR DE VIDEO (frame-por-frame + rastreador)
# ============================================================================
MIN_SCORE = 0.5
OBJ_COLORS = [(0, 200, 255), (255, 90, 90), (120, 220, 120), (255, 200, 60),
              (200, 120, 255), (255, 140, 200), (120, 200, 255), (180, 180, 120)]


def _png_b64(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _cdist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _color_hist(img, mask, bins=4):
    pix = img[mask]
    if len(pix) == 0:
        return np.zeros(bins ** 3, np.float32)
    q = (pix.astype(np.int32) * bins // 256).clip(0, bins - 1)
    idx = q[:, 0] * bins * bins + q[:, 1] * bins + q[:, 2]
    h = np.bincount(idx, minlength=bins ** 3).astype(np.float32)
    return h / (h.sum() + 1e-6)


def _hist_sim(h1, h2):
    return float(np.minimum(h1, h2).sum())


def _translate(text):
    """Traduce a ingles (auto-detecta idioma). Si falla, devuelve el original."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source="auto", target="en").translate(text) or text
    except Exception:
        return text


def _detect(image_pil, prompt):
    """Devuelve la lista de detecciones (multi-objeto) de UN frame."""
    img = np.array(image_pil)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = _processor.set_image(image_pil)
        out = _processor.set_text_prompt(state=state, prompt=prompt)
    masks, scores = out["masks"], out["scores"]
    dets = []
    for i in range(int(masks.shape[0])):
        sc = float(scores[i].float().cpu())
        if sc < MIN_SCORE:
            continue
        m = masks[i, 0].float().cpu().numpy()
        mb = m > (0.5 if m.max() <= 1.0 + 1e-3 else 0.0)
        if mb.sum() < 200:
            continue
        ys, xs = np.where(mb)
        dets.append({
            "mask": mb,
            "centroid": (float(xs.mean()), float(ys.mean())),
            "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
            "score": sc,
            "hist": _color_hist(img, mb),
            "label": prompt,
        })
    return dets


class Tracker:
    """Rastreador casero: empareja por IoU/centroide (con prediccion) + apariencia."""
    def __init__(self, w_img, match_thr=0.6, max_lost=30, w_motion=0.6):
        self.tracks, self.next_id = {}, 0
        self.w_img = w_img
        self.match_thr, self.max_lost, self.w_motion = match_thr, max_lost, w_motion

    def _predict(self, t):
        tr = t["trail"]
        if len(tr) >= 2:
            (x0, y0), (x1, y1) = tr[-2], tr[-1]
            return (x1 + (x1 - x0), y1 + (y1 - y0))
        return tr[-1]

    def update(self, dets):
        ids = list(self.tracks.keys())
        assign = {}
        if ids and dets:
            cost = np.ones((len(dets), len(ids)))
            for i, d in enumerate(dets):
                for j, tid in enumerate(ids):
                    t = self.tracks[tid]
                    motion = min(1 - _iou(d["bbox"], t["bbox"]),
                                 _cdist(d["centroid"], self._predict(t)) / (0.25 * self.w_img))
                    app = 1 - _hist_sim(d["hist"], t["hist"])
                    cost[i, j] = self.w_motion * motion + (1 - self.w_motion) * app
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] <= self.match_thr:
                    assign[r] = ids[c]
        seen, out = set(), []
        for i, d in enumerate(dets):
            if i in assign:
                tid = assign[i]
                t = self.tracks[tid]
                t.update(bbox=d["bbox"], centroid=d["centroid"], lost=0)
                t["trail"].append(d["centroid"])
                t["hist"] = 0.7 * t["hist"] + 0.3 * d["hist"]
            else:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {"bbox": d["bbox"], "centroid": d["centroid"],
                                    "trail": [d["centroid"]], "lost": 0, "hist": d["hist"]}
            seen.add(tid)
            out.append((tid, d, list(self.tracks[tid]["trail"])))
        for tid in ids:
            if tid not in seen:
                self.tracks[tid]["lost"] += 1
        for tid in [t for t in self.tracks if self.tracks[t]["lost"] > self.max_lost]:
            del self.tracks[tid]
        return out


def _extract_frames(video_bytes, max_frames=None, stride=1, target_fps=None, max_seconds=None):
    """Escribe el video a un temporal y extrae sus frames como PIL.
    Si se da target_fps, calcula el stride a partir de los fps reales del video.
    Si se da max_seconds, solo lee los primeros N segundos del video.
    Devuelve (frames, fps_efectivos)."""
    path = tempfile.mktemp(suffix=".mp4")
    with open(path, "wb") as f:
        f.write(video_bytes)
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if target_fps:
        stride = max(1, round(src_fps / float(target_fps)))
    max_src = int(max_seconds * src_fps) if max_seconds else None
    frames, i = [], 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        if max_src is not None and i >= max_src:
            break
        if i % max(1, stride) == 0:
            frames.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            if max_frames and len(frames) >= max_frames:
                break
        i += 1
    cap.release()
    try:
        os.remove(path)
    except OSError:
        pass
    eff_fps = round(src_fps / max(1, stride), 2)
    return frames, eff_fps


def process_video(frames, prompt, min_presence=0.25, export_path=None, fps_out=10):
    """Generador en streaming:
       'progress'      -> por cada frame mientras detecta+rastrea
       'result_meta'   -> las capas (objetos + miniaturas + rastros), una vez
       'frame'         -> overlay compuesto, uno por frame (se transmiten en orden)
       'result_done'   -> al terminar
    Si export_path se da, además escribe un mp4 compuesto (original + overlay).
    """
    N = len(frames)
    W, H = frames[0].size                       # PIL: (ancho, alto)
    tracker = Tracker(w_img=W)
    per_frame, obj_frames = [], defaultdict(list)

    # "robot pequeño, pelota naranja" -> ["robot pequeño", "pelota naranja"]
    originals = [p.strip() for p in prompt.split(",") if p.strip()] or [prompt]
    english = [_translate(p) for p in originals]      # SAM 3 trabaja en inglés
    yield {"type": "prompts", "pairs": [[o, e] for o, e in zip(originals, english)]}

    for fi, img in enumerate(frames):
        dets = []
        for orig, eng in zip(originals, english):
            for d in _detect(img, eng):              # detectar con el inglés
                d["label"] = orig                    # pero mostrar lo que escribiste
                dets.append(d)
        tracked = tracker.update(dets)
        per_frame.append(tracked)
        for tid, d, _ in tracked:
            obj_frames[tid].append((fi, d))
        yield {"type": "progress", "done": fi + 1, "total": N}

    # Objetos estables = las "capas"
    stable = sorted([t for t, fs in obj_frames.items() if len(fs) >= min_presence * N])
    color_of = {t: OBJ_COLORS[i % len(OBJ_COLORS)] for i, t in enumerate(stable)}

    objects = []
    for tid in stable:
        fi, d = max(obj_frames[tid], key=lambda x: x[1]["mask"].sum())
        img = np.array(frames[fi])
        m = d["mask"]
        ys, xs = np.where(m)
        cut = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        cm = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        rgba = np.zeros((cut.shape[0], cut.shape[1], 4), np.uint8)
        rgba[..., :3] = cut
        rgba[..., 3] = np.where(cm, 255, 0)
        objects.append({
            "id": int(tid),
            "color": list(color_of[tid]),
            "label": d.get("label", prompt),
            "trail": [[int(fi), float(dd["centroid"][0]), float(dd["centroid"][1])]
                      for fi, dd in obj_frames[tid]],
            "thumb_png": _png_b64(rgba),
        })

    # Primero la metadata (capas) -> el frontend ya puede pintar el panel
    yield {"type": "result_meta", "n_frames": N, "width": int(W), "height": int(H),
           "objects": objects}

    # Escritor de mp4 (opcional) para la descarga
    writer = None
    if export_path:
        writer = cv2.VideoWriter(export_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 float(fps_out) if fps_out else 10.0, (W, H))

    # Luego, por frame: el original + las máscaras POR CAPA (id, caja, máscara)
    sset = set(stable)
    for fi, tracked in enumerate(per_frame):
        ov = np.zeros((H, W, 4), np.uint8)   # solo para el mp4 de descarga
        objs = []
        for tid, d, _ in tracked:
            if tid in sset:
                r, g, b = color_of[tid]
                ov[d["mask"]] = [r, g, b, 130]
                # máscara individual: blanca donde está el objeto, transparente el resto
                mh = np.zeros((H, W, 4), np.uint8)
                mh[d["mask"]] = [255, 255, 255, 255]
                objs.append({
                    "id": int(tid),
                    "bbox": [int(x) for x in d["bbox"]],
                    "mask_png": _png_b64(mh),
                })

        jb = io.BytesIO()
        frames[fi].save(jb, format="JPEG", quality=70)
        base_jpg = base64.b64encode(jb.getvalue()).decode("ascii")
        yield {"type": "frame", "frame": fi, "base_jpg": base_jpg, "objects": objs}

        if writer is not None:
            base = np.array(frames[fi])
            a = ov[..., 3:4].astype(np.float32) / 255.0
            comp = (base * (1 - a) + ov[..., :3] * a).astype(np.uint8)
            writer.write(cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))

    if writer is not None:
        writer.release()

    yield {"type": "result_done"}


# ============================================================================
#  App FastAPI + WebSocket
# ============================================================================
app = FastAPI(title="SAM3 Backend")


@app.get("/")
def health():
    return {"status": "ok"}


async def _run_video(websocket, video_bytes, prompt, mp, max_frames, stride, target_fps, max_seconds=None):
    """Extrae frames, procesa en un hilo y transmite progreso + resultado en streaming.
    Devuelve la ruta del mp4 compuesto (para descarga)."""
    frames, eff_fps = _extract_frames(video_bytes, max_frames=max_frames,
                                      stride=stride, target_fps=target_fps,
                                      max_seconds=max_seconds)
    if not frames:
        await websocket.send_text(json.dumps({"type": "error", "msg": "no se pudieron leer frames"}))
        return None

    # Avisar cuántos frames y a qué fps, antes de empezar el trabajo pesado
    await websocket.send_text(json.dumps(
        {"type": "video_info", "n_frames": len(frames), "proc_fps": eff_fps}))

    export_path = tempfile.mktemp(suffix=".mp4")

    # Procesamiento pesado en un hilo aparte para NO bloquear el event loop.
    # El hilo empuja cada update a una cola; aqui los enviamos al cliente.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for update in process_video(frames, prompt, mp,
                                        export_path=export_path, fps_out=eff_fps):
                loop.call_soon_threadsafe(queue.put_nowait, update)
        except Exception as e:
            loop.call_soon_threadsafe(queue.put_nowait, {"type": "error", "msg": str(e)})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    loop.run_in_executor(None, _worker)
    while True:
        update = await queue.get()
        if update is None:
            break
        if update.get("type") == "result_meta":
            update["proc_fps"] = eff_fps
        await websocket.send_text(json.dumps(update))

    return export_path


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    current_image = None
    video_chunks = []        # buffer para carga por pedacitos
    video_meta = {}
    last_export = None       # ruta del ultimo mp4 compuesto
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            t = msg.get("type")

            if t == "image":
                raw = base64.b64decode(msg["data"])
                current_image = Image.open(io.BytesIO(raw)).convert("RGB")
                W, H = current_image.size
                await websocket.send_text(json.dumps({"type": "image_ready", "width": W, "height": H}))

            elif t == "text":
                if current_image is None:
                    await websocket.send_text(json.dumps({"type": "error", "msg": "primero carga una imagen"}))
                    continue
                b64, n, score = segmentar(current_image, msg.get("prompt", ""))
                W, H = current_image.size
                await websocket.send_text(json.dumps({
                    "type": "mask", "prompt": msg.get("prompt", ""),
                    "n": n, "score": round(score, 4), "width": W, "height": H, "png": b64}))

            # --- Carga de video en UN mensaje (para pruebas en Colab) -------
            elif t == "video":
                raw = base64.b64decode(msg["data"])
                last_export = await _run_video(websocket, raw, msg.get("prompt", ""),
                                 float(msg.get("min_presence", 0.25)),
                                 msg.get("max_frames"), int(msg.get("stride", 1)),
                                 msg.get("target_fps"), msg.get("max_seconds"))

            # --- Carga de video por PEDACITOS (chunks) — para Flutter -------
            elif t == "video_start":
                video_chunks = []
                video_meta = {
                    "prompt": msg.get("prompt", ""),
                    "min_presence": float(msg.get("min_presence", 0.25)),
                    "max_frames": msg.get("max_frames"),
                    "stride": int(msg.get("stride", 1)),
                    "target_fps": msg.get("target_fps"),
                    "max_seconds": msg.get("max_seconds"),
                }
                await websocket.send_text(json.dumps({"type": "video_ack"}))

            elif t == "video_chunk":
                video_chunks.append(msg["data"])

            elif t == "video_end":
                raw = base64.b64decode("".join(video_chunks))
                video_chunks = []
                last_export = await _run_video(websocket, raw, video_meta.get("prompt", ""),
                                 video_meta.get("min_presence", 0.25),
                                 video_meta.get("max_frames"),
                                 video_meta.get("stride", 1),
                                 video_meta.get("target_fps"),
                                 video_meta.get("max_seconds"))

            # --- Descargar el mp4 compuesto (por chunks) --------------------
            elif t == "export_video":
                if not last_export or not os.path.exists(last_export):
                    await websocket.send_text(json.dumps({"type": "error", "msg": "procesa un video primero"}))
                    continue
                with open(last_export, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("ascii")
                await websocket.send_text(json.dumps({"type": "export_start", "size": len(b64)}))
                CH = 256 * 1024
                for i in range(0, len(b64), CH):
                    await websocket.send_text(json.dumps(
                        {"type": "export_chunk", "data": b64[i:i + CH]}))
                await websocket.send_text(json.dumps({"type": "export_end"}))

            else:
                await websocket.send_text(json.dumps({"type": "error", "msg": "tipo no soportado"}))

    except WebSocketDisconnect:
        print("Cliente desconectado")
