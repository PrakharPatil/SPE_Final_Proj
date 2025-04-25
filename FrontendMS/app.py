# from fastapi import FastAPI, Form, Request
# from fastapi.responses import HTMLResponse
# from fastapi.templating import Jinja2Templates
# import joblib
# import pandas as pd
#
# # Load the pre-trained model pipeline (ensure it contains preprocessing steps)
# model = joblib.load("../TransformerMS/random_forest_pipeline.joblib")
#
# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
#
# @app.get("/", response_class=HTMLResponse)
# async def form_page(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
#
# @app.post("/predict", response_class=HTMLResponse)
# async def predict(
#     request: Request,
#     area: int = Form(...),
#     bedrooms: int = Form(...),
#     bathrooms: int = Form(...),
#     area_type: int = Form(...),
#     city: int = Form(...),
#     furnishing: int = Form(...),
#     tenant: int = Form(...),
#     contact: int = Form(...),
#     posted_month: int = Form(...),
#     posted_year: int = Form(...),
#     current_floor: int = Form(...),
#     total_floor: int = Form(...),
#     area_te: float = Form(...)
# ):
#     # Construct DataFrame with expected feature names
#     input_data = pd.DataFrame([{
#         'BHK': bedrooms,
#         'Size': area,
#         'Area Type': area_type,
#         'City': city,
#         'Furnishing Status': furnishing,
#         'Tenant Preferred': tenant,
#         'Bathroom': bathrooms,
#         'Point of Contact': contact,
#         'Posted_Month': posted_month,
#         'Posted_Year': posted_year,
#         'Current_Floor': current_floor,
#         'Total_Floor': total_floor,
#         'Area_Locality_TE': area_te
#     }])
#
#     try:
#         # Predict using model (ensure model handles necessary preprocessing internally)
#         predicted_rent = model.predict(input_data)[0]
#
#         return templates.TemplateResponse("result.html", {
#             "request": request,
#             "predicted_rent": f"₹{int(predicted_rent):,}"
#         })
#     except Exception as e:
#         # Handle any errors during prediction and return a user-friendly message
#         return templates.TemplateResponse("result.html", {
#             "request": request,
#             "predicted_rent": "Error in prediction. Please try again later."
#         })
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os

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

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})