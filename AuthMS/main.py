import uvicorn
from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from AuthMS.database.database import engine, SessionLocal
import auth, database.database
from AuthMS.schema import schemas
from AuthMS.model import models

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register", response_model=schemas.UserResponse)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Check if username already exists
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail="Username already registered")
    # Hash the password and save the new user
    hashed_password = auth.get_password_hash(user.password)
    db_user = models.User(username=user.username, hashed_password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login", response_model=schemas.Token)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    # Verify user exists and password is correct
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not auth.verify_password(user.password, db_user.hashed_password):
        # Invalid credentials
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid username or password")
    # Create tokens
    access_token = auth.create_access_token(data={"sub": db_user.username})
    refresh_token = auth.create_refresh_token(data={"sub": db_user.username})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}

@app.post("/refresh", response_model=schemas.Token)
def refresh_token(token_refresh: schemas.TokenRefresh):
    try:
        # Decode refresh token
        payload = jwt.decode(token_refresh.refresh_token, auth.SECRET_KEY, algorithms=[auth.ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    # Generate new tokens for the user
    access_token = auth.create_access_token(data={"sub": username})
    refresh_token = auth.create_refresh_token(data={"sub": username})
    return {"access_token": access_token, "refresh_token": refresh_token, "token_type": "bearer"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="localhost", port=8082, reload=True)