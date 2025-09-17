# db.py
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DB_PATH = os.getenv("DB_PATH", "./controller.db")
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

# Engine & session
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class Worker(Base):
    __tablename__ = "workers"
    id = Column(Integer, primary_key=True, index=True)
    worker_id = Column(String, unique=True, index=True, nullable=False)
    model = Column(String, index=True, nullable=False)
    endpoint = Column(String, nullable=False)   # public URL for worker e.g. https://abc.salad.run
    status = Column(String, default="ready")    # ready, busy, offline
    container_id = Column(String, nullable=True)
    last_heartbeat = Column(DateTime, default=datetime.timezone.utc)
    last_active = Column(DateTime, default=datetime.timezone.utc)
    info = Column(JSON, nullable=True)

class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True, nullable=False)
    model = Column(String, index=True, nullable=False)
    prompt = Column(Text, nullable=True)
    params = Column(JSON, nullable=True)
    worker_id = Column(String, nullable=True, index=True)   # assigned worker id
    status = Column(String, default="queued")  # queued, running, completed, failed
    result_r2_key = Column(String, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.timezone.utc)
    updated_at = Column(DateTime, default=datetime.timezone.utc)

class Device(Base):
    __tablename__ = "devices"
    id = Column(Integer, primary_key=True, index=True)
    device_uuid = Column(String, unique=False, index=True, nullable=True)
    fingerprint_hash = Column(String, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.timezone.utc)
    last_seen = Column(DateTime, default=datetime.timezone.utc)
    request_count = Column(Integer, default=0)
    banned = Column(Boolean, default=False)
    meta = Column(JSON, nullable=True)

def init_db():
    Base.metadata.create_all(bind=engine)