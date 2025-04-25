# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi import Request
# import httpx
# import asyncio
 
# app = FastAPI()
 
# # Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
 
# connected_clients = []
 
# @app.websocket("/ws/stream")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     connected_clients.append(websocket)
 
#     try:
#         while True:
#             await asyncio.sleep(1)  # Keep connection alive
#     except WebSocketDisconnect:
#         connected_clients.remove(websocket)
# from datetime import datetime
# @app.post("/api/prompt")
# async def receive_prompt(request: Request):
#     data = await request.json()
#     prompt = data.get("prompt")
#     print("Received prompt:", prompt)
    
#     try:
#         async with httpx.AsyncClient(timeout=None) as client:
#             async with client.stream(
#                 "POST",
#                 "http://172.31.76.47:8001/generate/",
#                 json={"prompt": prompt}
#             ) as response:
#                 if response.status_code != 200:
#                     print("Error from Django:", response.status_code)
#                     return {"status": "error", "message": "Model server error"}
 
#                 # IMPORTANT: stream and forward token by token
#                 async for line in response.aiter_lines():
#                     if line.strip():
#                         print(f"fasta api Streaming token:{datetime.now()}" , line)
#                         # Forward to all connected WebSocket clients
#                         for ws in connected_clients:
#                             try:
#                                 await ws.send_text(line)
#                             except Exception as e:
#                                 print(f"WebSocket send error: {e}")
#         return {"status": "streaming completed"}
 
#     except Exception as e:
#         print("Error:", e)
#         return {"status": "error", "message": str(e)}
 
# # Run FastAPI
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="172.31.76.47", port=8002)
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
import httpx
import asyncio
from datetime import datetime

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # secure in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected clients by client_id
connected_clients = {}

@app.websocket("/ws/stream/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    connected_clients[client_id] = websocket
    print(f"Client connected: {client_id}")
    try:
        while True:
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        print(f"Client disconnected: {client_id}")
        connected_clients.pop(client_id, None)

@app.post("/api/prompt/{client_id}")
async def receive_prompt(client_id: str, request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    print("Received prompt:", prompt)

    websocket = connected_clients.get(client_id)
    if not websocket:
        return {"status": "error", "message": "WebSocket not connected"}

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://172.31.76.47:8001/generate/",
                json={"prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    return {"status": "error", "message": "Model server error"}

                async for line in response.aiter_lines():
                    if line.strip():
                        print(f"Streaming to {client_id} at {datetime.now()}: {line}")
                        try:
                            await websocket.send_text(line)
                        except Exception as e:
                            print(f"Send error to {client_id}: {e}")
        return {"status": "streaming completed"}

    except Exception as e:
        print("Error:", e)
        return {"status": "error", "message": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.31.76.47", port=8002)