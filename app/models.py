"""
Database Models for LexDocuAI

This module defines the SQLAlchemy models for the LexDocuAI application.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, ForeignKey, Table, Boolean
from sqlalchemy.orm import declarative_base, relationship
import datetime

Base = declarative_base()

class User(Base):
    """User model for authentication and access control."""
    
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    role = Column(String(20), default='user')  # 'user', 'admin', etc.
    
    documents = relationship("Document", back_populates="owner")
    
    def __repr__(self):
        return f"<User {self.username}>"

class Document(Base):
    """Document model to store uploaded legal documents."""
    
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False)
    document_type = Column(String(50))
    file_path = Column(String(255))
    file_size = Column(Integer)
    mime_type = Column(String(100))
    upload_date = Column(DateTime, default=datetime.datetime.utcnow)
    
    owner_id = Column(Integer, ForeignKey('users.id'))
    owner = relationship("User", back_populates="documents")
    analysis = relationship("DocumentAnalysis", back_populates="document", uselist=False, cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document {self.original_filename}>"

class DocumentAnalysis(Base):
    """Stores the analysis results for a document."""
    
    __tablename__ = 'document_analysis'
    
    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey('documents.id'), unique=True, nullable=False)
    analysis_date = Column(DateTime, default=datetime.datetime.utcnow)
    summary = Column(Text)
    risk_assessment = Column(Text)
    processing_time = Column(Float)
    models_used = Column(Text) # Store as JSON string

    clauses = relationship("DocumentClause", back_populates="analysis", cascade="all, delete-orphan")
    parties = relationship("DocumentParty", back_populates="analysis", cascade="all, delete-orphan")
    
    document = relationship("Document", back_populates="analysis")
    
    def __repr__(self):
        return f"<DocumentAnalysis for doc_id={self.document_id}>"

class DocumentClause(Base):
    """Stores information about clauses extracted from documents."""
    
    __tablename__ = 'document_clauses'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('document_analysis.id'), nullable=False)
    title = Column(String(255))
    content = Column(Text)
    
    analysis = relationship("DocumentAnalysis", back_populates="clauses")
    
    def __repr__(self):
        return f"<DocumentClause {self.title}>"

class DocumentParty(Base):
    """Stores information about parties in legal documents."""
    
    __tablename__ = 'document_parties'
    
    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey('document_analysis.id'), nullable=False)
    name = Column(String(255))
    party_type = Column(String(50))
    
    analysis = relationship("DocumentAnalysis", back_populates="parties")
    
    def __repr__(self):
        return f"<DocumentParty {self.name}>"
