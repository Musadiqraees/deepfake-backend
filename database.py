from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

# SQLite database file
SQLALCHEMY_DATABASE_URL = "sqlite:///./jobs.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class JobHistory(Base):
    __tablename__ = "job_history"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="processing")
    result = Column(String, nullable=True)
    confidence = Column(Float, nullable=True)
    thumbnail_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Create tables immediately
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()