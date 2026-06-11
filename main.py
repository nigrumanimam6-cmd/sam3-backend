# ============================================================================
#  SAM3 BACKEND · main.py
#  FastAPI + WebSocket. Ahora segmenta la imagen que ENVÍA el cliente
#  (Flutter), no una fija. La imagen se manda UNA vez por carga y se cachea
#  en la conexión; cada prompt de texto la reutiliza.
#
#  Protocolo WebSocket:
#    Cliente -> {"type": "image", "data": "<png/jpg en base64>"}
#    Servidor-> {"type": "image_ready", "width": W, "height": H}
#    Cliente -> {"type": "text", "prompt": "..."}
#    Servidor-> {"type": "mask", "n":N, "score":S, "width":W, "height":H, "png":"<base64>"}
# ============================================================================

import io
import base64
import json

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- Estado global: el modelo (compartido por todas las conexiones) ---------
_processor = None


def load_model():
    """Carga SAM 3. Llamar UNA vez al arrancar (desde el launcher de Colab)."""
    global _processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()      # descarga los pesos de HF la 1ª vez
    _processor = Sam3Processor(model)
    return _processor


def segmentar(image, prompt: str):
    """Segmenta `image` (PIL) con un prompt de texto.
    Devuelve (png_base64_overlay, n_objetos, score)."""
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
    rgba[m_bin] = [0, 200, 255, 130]      # cian semitransparente

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, n, float(scores[0].float().cpu())


# --- App FastAPI ------------------------------------------------------------
app = FastAPI(title="SAM3 Backend")


@app.get("/")
def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    current_image = None          # imagen cacheada de ESTA conexión (PIL)
    try:
        while True:
            msg = json.loads(await websocket.receive_text())
            t = msg.get("type")

            if t == "image":
                # El cliente nos manda su imagen (base64). La decodificamos
                # y la guardamos; aquí no segmentamos todavía.
                raw = base64.b64decode(msg["data"])
                current_image = Image.open(io.BytesIO(raw)).convert("RGB")
                W, H = current_image.size
                await websocket.send_text(json.dumps({
                    "type": "image_ready", "width": W, "height": H,
                }))

            elif t == "text":
                if current_image is None:
                    await websocket.send_text(json.dumps(
                        {"type": "error", "msg": "primero carga una imagen"}))
                    continue
                b64, n, score = segmentar(current_image, msg.get("prompt", ""))
                W, H = current_image.size
                await websocket.send_text(json.dumps({
                    "type": "mask",
                    "prompt": msg.get("prompt", ""),
                    "n": n,
                    "score": round(score, 4),
                    "width": W,
                    "height": H,
                    "png": b64,
                }))

            else:
                await websocket.send_text(json.dumps(
                    {"type": "error", "msg": "tipo no soportado"}))

    except WebSocketDisconnect:
        print("Cliente desconectado")
