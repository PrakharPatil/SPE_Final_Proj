from fastapi import FastAPI
from pydantic import BaseModel
from Model.model_utils import ModelLoader
import uvicorn

app = FastAPI()
model_loader = ModelLoader()

class GenerationRequest(BaseModel):
    prompt: str

class GenerationResponse(BaseModel):
    generated_text: str

@app.post("/generate", response_model=GenerationResponse)
async def generate_text(request: GenerationRequest):
    generated = model_loader.generate_response(
        prompt=request.prompt
    )
    return {"generated_text": generated}


# uvicorn.run(app, host="127.0.0.1", port=8082)