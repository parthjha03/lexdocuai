import os

# Get the absolute path of the directory where this file is located (the 'app' directory)
basedir = os.path.abspath(os.path.dirname(__file__))

class Config:
    # --- IMPORTANT: Use environment variables for all sensitive/config values ---

    # Flask and DB Configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "default-secret-key")
    SQLALCHEMY_DATABASE_URI = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///instance/lexdocuai.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = os.getenv("SQLALCHEMY_TRACK_MODIFICATIONS", "False") == "True"
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", os.path.join(basedir, '..', 'uploads'))
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 50 * 1024 * 1024))  # 50MB default

    # LLM and Vector DB API Keys
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    DEEPINFRA_API_KEY = os.getenv('DEEPINFRA_API_KEY')
    PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
    PINECONE_INDEX_NAME = os.getenv('PINECONE_INDEX_NAME', "lexdocuai-index")
    
    # Pinecone Config
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 768)) # For BAAI/bge-base-en-v1.5
    PINECONE_METRIC = os.getenv('PINECONE_METRIC', 'cosine')
    PINECONE_CLOUD = os.getenv('PINECONE_CLOUD', 'aws')
    PINECONE_REGION = os.getenv('PINECONE_REGION', 'us-east-1')