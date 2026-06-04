from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="SAM3 Backend")

@app.get("/")
def health():
    return {"status": "ok", "message": "SAM3 backend vivo 🚀"}

@app.websocket("/ws")
async def websocket_echo(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            data = await ws.receive_text()
            await ws.send_text(f"ECO: {data}")
    except WebSocketDisconnect:
        print("Cliente desconectado")
