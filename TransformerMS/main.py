from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware

from Model.model_utils import ModelLoader
import uvicorn

app = FastAPI()
model_loader = ModelLoader()

class GenerationRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    generated_text: str
# In backend's main.py
from fastapi.middleware.cors import CORSMiddleware


app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Update your endpoint to expect JSON
class GenerationRequest(BaseModel):
    prompt: str

class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000,
                       example="What is artificial intelligence?")
@app.post("/generate")
async def generate_text(request: GenerationRequest):  # Changed from Form
    generated = model_loader.generate_response(request.prompt)
    return {"generated_text": generated}
if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8081, reload=True)

# uvicorn.run(app, host="127.0.0.1", port=8082)