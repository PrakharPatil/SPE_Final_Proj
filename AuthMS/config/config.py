import os
from datetime import timedelta

from dotenv import load_dotenv
from pydantic.v1 import BaseSettings

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
env_path = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=env_path)

class Settings(BaseSettings):
    DB_HOST: str
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str
    SECRET_KEY: str
    JWT_ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int

    class Config:
        env_file = os.path.join(BASE_DIR, '.env')
        # env_file = ".env"

settings = Settings()
# JWT settings (use strong secret in production, e.g. from an environment variable)
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "my-super-secret-key")  # Replace with a secure random key
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = 15  # Short-lived access token
REFRESH_TOKEN_EXPIRE_DAYS = 7    # Longer-lived refresh token (e.g. 7 days)

# Database URL (configure with your MySQL credentials)
DATABASE_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}/{settings.DB_NAME}"
