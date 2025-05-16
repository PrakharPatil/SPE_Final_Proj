import httpx
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from pydantic import BaseModel
from fastapi import Body  # Add this import


app = FastAPI()

# Base directory configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # FrontendMS/my-website

# Template configuration
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.loader.searchpath.append(os.path.join(BASE_DIR, "components"))  # Add components

# Mount static directories
app.mount("/styles", StaticFiles(directory=os.path.join(BASE_DIR, "styles")), name="styles")
app.mount("/assets", StaticFiles(directory=os.path.join(BASE_DIR, "assets")), name="assets")
app.mount("/scripts", StaticFiles(directory=os.path.join(BASE_DIR, "scripts")), name="scripts")

# Backend service URL (update port if different)
# BACKEND_URL = "http://0.0.0.0:8082/generate"
BACKEND_URL = "http://transformer-container:8082/generate"
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


# Add this model definition
class GenerationRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def proxy_generation(request_data: GenerationRequest = Body(...)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                BACKEND_URL,
                json={"prompt": request_data.prompt}
            )
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8081, reload=True)