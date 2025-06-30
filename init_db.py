import os
from sqlalchemy import create_engine
from app.models import Base
from config import Config

def init_database():
    """Creates the database tables."""
    print("Initializing database...")
    engine = create_engine(Config.SQLALCHEMY_DATABASE_URI)
    Base.metadata.create_all(engine)
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_database()
