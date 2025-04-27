# In the main application startup code (see below)
from database import engine, Base
from AuthMS.model.models import User

Base.metadata.create_all(bind=engine)
