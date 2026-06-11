# ============================================================================
#  SAM3 BACKEND · main.py
#  FastAPI + WebSocket que segmenta una imagen con SAM 3 a partir de un
#  prompt de TEXTO y devuelve la máscara como overlay PNG (RGBA, base64).
#
#  Este archivo vive en el repo (GitHub). Colab solo lo clona y lo ejecuta.
#  La autenticación (HF / ngrok) y el túnel se manejan en el launcher de Colab,
#  NO aquí — así no se filtran secretos al repo.
# ============================================================================

import io
import base64
import json

import numpy as np
import torch
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

# --- Estado global del modelo (se llena al llamar load_model) ---------------
_processor = None
_test_image = None


def load_model(test_image_path: str = "/content/sam3/assets/images/truck.jpg"):
    """Carga SAM 3 y la imagen de prueba. Llamar UNA sola vez al arrancar.
    Los imports de sam3 van aquí dentro para que `import main` sea ligero
    y para que la ruta de sam3 ya esté lista cuando se ejecute."""
    global _processor, _test_image
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model()          # descarga los pesos de HF la 1ª vez
    _processor = Sam3Processor(model)
    _test_image = Image.open(test_image_path).convert("RGB")
    return _processor


def segmentar(prompt: str):
    """Corre SAM 3 sobre la imagen de prueba con un prompt de texto.
    Devuelve (png_base64, n_objetos, score) — la máscara como overlay RGBA."""
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = _processor.set_image(_test_image)
        out = _processor.set_text_prompt(state=state, prompt=prompt)

    masks, scores = out["masks"], out["scores"]
    n = int(masks.shape[0])
    if n == 0:
        return None, 0, 0.0

    m = masks[0, 0].float().cpu().numpy()             # (alto, ancho)
    thr = 0.5 if m.max() <= 1.0 + 1e-3 else 0.0       # umbral robusto
    m_bin = m > thr

    # Overlay RGBA: cian semitransparente sobre el objeto, fondo transparente
    H, W = m_bin.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    rgba[m_bin] = [0, 200, 255, 130]

    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return b64, n, float(scores[0].float().cpu())


# --- App FastAPI ------------------------------------------------------------
app = FastAPI(title="SAM3 Backend")


@app.get("/")
def health():
    return {"status": "ok", "image": _test_image.size if _test_image else None}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            msg = json.loads(await websocket.receive_text())

            if msg.get("type") == "text":
                prompt = msg.get("prompt", "")
                b64, n, score = segmentar(prompt)
                W, H = _test_image.size
                await websocket.send_text(json.dumps({
                    "type": "mask",
                    "prompt": prompt,
                    "n": n,
                    "score": round(score, 4),
                    "width": W,
                    "height": H,
                    "png": b64,
                }))
            else:
                await websocket.send_text(json.dumps(
                    {"type": "error", "msg": "tipo no soportado"}
                ))

    except WebSocketDisconnect:
        print("Cliente desconectado")
