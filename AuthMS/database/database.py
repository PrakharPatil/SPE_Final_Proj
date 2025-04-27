from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from AuthMS.config.config import DATABASE_URL


# Create the SQLAlchemy engine (replace driver and credentials as needed)
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

# SessionLocal is a factory for new SQLAlchemy sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for our models
Base = declarative_base()
