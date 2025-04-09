from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    area: int = Form(...),
    bedrooms: int = Form(...),
    bathrooms: int = Form(...)
):
    return templates.TemplateResponse("result.html", {
        "request": request,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms
    })