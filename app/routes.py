from flask import Blueprint, render_template, request, jsonify, current_app
import os
import uuid
import json
import hashlib
from .models import Document, DocumentAnalysis, DocumentClause, DocumentParty
from .services.legal_analyzer import EnhancedLegalDocumentAnalyzer
from .services.rag_service import get_rag_service
import logging
from .services.rag_service import get_rag_service

bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@bp.route('/')
def index():
    session = current_app.db_session
    documents = session.query(Document).order_by(Document.upload_date.desc()).all()
    return render_template('index.html', documents=documents)

@bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Calculate file hash
        file_content = file.read()
        file.seek(0) # Reset file pointer
        file_hash = hashlib.sha256(file_content).hexdigest()

        # Check for duplicates and handle them by replacing the old file
        session = current_app.db_session
        existing_doc = session.query(Document).filter_by(file_hash=file_hash).first()
        if existing_doc:
            logger.info(f"Duplicate file detected: '{existing_doc.original_filename}'. Deleting old entry to replace it.")
            
            # 1. Delete from RAG service (Pinecone)
            rag_service = get_rag_service()
            rag_service.delete_document(existing_doc.id)
            
            # 2. Delete the physical file
            try:
                if os.path.exists(existing_doc.file_path):
                    os.remove(existing_doc.file_path)
                    logger.info(f"Deleted old file from disk: {existing_doc.file_path}")
            except OSError as e:
                logger.error(f"Error deleting file {existing_doc.file_path}: {e}")

            # 3. Delete from database
            session.delete(existing_doc)
            session.commit()
            logger.info(f"Deleted old document record from database: ID {existing_doc.id}")

        # Save the file temporarily for analysis
        original_filename = file.filename
        ext = os.path.splitext(original_filename)[1].lower()
        unique_filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], unique_filename)
        with open(file_path, 'wb') as f:
            f.write(file_content)
        file_size = len(file_content)

        # Analyze the document
        analyzer = EnhancedLegalDocumentAnalyzer()
        is_legal = analyzer.is_legal_document(file_path)
        if not is_legal:
            os.remove(file_path) # Clean up the temporary file
            return jsonify({"error": "The uploaded file does not appear to be a legal document."}), 400

        analysis_data = analyzer.analyze_document(file_path)

        if "error" in analysis_data:
            return jsonify(analysis_data), 500

        # Save to database
        session = current_app.db_session
        new_doc = Document(
            filename=unique_filename,
            original_filename=original_filename,
            file_hash=file_hash,
            document_type=analysis_data['document_type'],
            file_path=file_path,
            file_size=file_size,
            mime_type=file.content_type
        )
        
        new_analysis = DocumentAnalysis(
            summary=analysis_data['summary'],
            risk_assessment=analysis_data['risk_assessment'],
            processing_time=analysis_data['processing_time'],
            models_used=json.dumps(analysis_data['models_used'])
        )

        for clause_data in analysis_data.get('clauses', []):
            new_analysis.clauses.append(DocumentClause(**clause_data))
        
        for party_data in analysis_data.get('parties', []):
            new_analysis.parties.append(DocumentParty(**party_data))

        new_doc.analysis = new_analysis
        session.add(new_doc)
        session.commit()

        # Index for RAG
        rag_service = get_rag_service()
        rag_service.index_document(new_doc.id, analysis_data['full_text'])

        return jsonify({"message": "File uploaded and analyzed successfully", "document_id": new_doc.id})

    return jsonify({"error": "File upload failed"}), 500

@bp.route('/document/<int:document_id>')
def view_document(document_id):
    session = current_app.db_session
    doc = session.query(Document).get(document_id)
    if not doc:
        return "Document not found", 404

    # Get the latest document to conditionally show the chat
    latest_doc = session.query(Document).order_by(Document.upload_date.desc()).first()
    is_latest = (latest_doc.id == doc.id) if latest_doc else False

    return render_template('document.html', document=doc, is_latest_document=is_latest)

@bp.route('/chat/<int:document_id>', methods=['POST'])
def chat(document_id):
    data = request.json
    user_question = data.get('message')
    if not user_question:
        return jsonify({"error": "No message provided"}), 400

    rag_service = get_rag_service()
    response = rag_service.chat_with_document(document_id, user_question)
    
    return jsonify({'response': response})
