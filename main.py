from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
import httpx
import asyncio
 
app = FastAPI()
 
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
connected_clients = []
 
@app.websocket("/ws/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
 
    try:
        while True:
            await asyncio.sleep(1)  # Keep connection alive
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
 
@app.post("/api/prompt")
async def receive_prompt(request: Request):
    data = await request.json()
    prompt = data.get("prompt")
    print("Received prompt:", prompt)
    
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                "http://172.31.76.47:8001/generate/",
                json={"prompt": prompt}
            ) as response:
                if response.status_code != 200:
                    print("Error from Django:", response.status_code)
                    return {"status": "error", "message": "Model server error"}
 
                # IMPORTANT: stream and forward token by token
                async for line in response.aiter_lines():
                    if line.strip():
                        print("fasta api Streaming token:", line)
                        # Forward to all connected WebSocket clients
                        for ws in connected_clients:
                            try:
                                await ws.send_text(line)
                            except Exception as e:
                                print(f"WebSocket send error: {e}")
        return {"status": "streaming completed"}
 
    except Exception as e:
        print("Error:", e)
        return {"status": "error", "message": str(e)}
 
# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="172.31.76.47", port=8002)