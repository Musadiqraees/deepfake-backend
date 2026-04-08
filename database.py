from sqlalchemy import create_engine, Column, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

SQLALCHEMY_DATABASE_URL = "sqlite:///./jobs.db"

# Vital for SQLite to work with FastAPI background tasks
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class JobHistory(Base):
    __tablename__ = "job_history"
    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="processing")  # 'processing', 'completed', 'error'
    result = Column(String, default="Analyzing...")
    confidence = Column(Float, default=0.0)
    thumbnail_path = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# Initialize table
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()