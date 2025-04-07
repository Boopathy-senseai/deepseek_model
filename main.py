from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # 🔥 Add this
from pydantic import BaseModel
import requests
from fastapi.responses import JSONResponse
 
app = FastAPI()
 
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can set specific domains instead of "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# Global variable to hold the system prompt
system_prompt = "You are a helpful AI assistant"
 
# Request models
class PromptRequest(BaseModel):
    prompt: str
 
class QueryRequest(BaseModel):
    query: str
 
@app.post("/set_prompt")
def set_prompt(data: PromptRequest):
    global system_prompt
    system_prompt = data.prompt
    return {"message": "Prompt updated successfully"}
 
@app.post("/ask")
def ask_model(data: QueryRequest):
    payload = {
        "prompt": data.query,
        "system_prompt": system_prompt,
        "temperature": 0.7
    }
 
    response = requests.post(
        "http://54.205.18.193:8000/generate/",
        headers={"Content-Type": "application/json"},
        json=payload
    )
 
    return response.json()
