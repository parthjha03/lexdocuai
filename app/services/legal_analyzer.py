"""
LexDocuAI Legal Document Analyzer
"""

import os
import re
import logging
import time
import spacy
import PyPDF2
import docx
from typing import List, Dict, Any

from .llm_service import get_llm_service, LLMCapability, LEGAL_SYSTEM_PROMPTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logger.warning("spaCy model not found. Downloading en_core_web_sm...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class EnhancedLegalDocumentAnalyzer:
    def __init__(self):
        self.llm_service = get_llm_service()
        logger.info("Enhanced Legal Document Analyzer initialized.")

    def is_legal_document(self, file_path: str) -> bool:
        """Check if the document is a legal document using an LLM call."""
        text = self._extract_text(file_path)
        if not text.strip():
            return False

        try:
            result = self.llm_service.generate(
                prompt=f"Is this a legal document? {text[:2000]}",
                capability=LLMCapability.CLASSIFICATION,
                system_prompt=LEGAL_SYSTEM_PROMPTS["legal_validation"],
                temperature=0.1
            )
            response = result.get("response", "").strip().lower()
            logger.info(f"Legal validation response: '{response}'")
            return response == 'yes'
        except Exception as e:
            logger.error(f"Error during legal document validation: {e}")
            return False

    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        start_time = time.time()
        text = self._extract_text(file_path)
        if not text:
            return {"error": "Could not extract text from document."}

        models_used = {}

        # 1. Classify Document Type
        doc_type_result = self.llm_service.generate(
            prompt=f"Analyze this document excerpt and classify its legal document type: {text[:1500]}",
            capability=LLMCapability.CLASSIFICATION,
            system_prompt=LEGAL_SYSTEM_PROMPTS["document_classification"],
            temperature=0.2
        )
        models_used["document_type"] = doc_type_result["provider"]
        doc_type = doc_type_result["response"].strip()

        # 2. Generate Summary
        summary_result = self.llm_service.generate(
            prompt=f"Summarize the following legal document: {text[:8000]}",
            capability=LLMCapability.SUMMARIZATION,
            system_prompt=LEGAL_SYSTEM_PROMPTS["document_summary"],
            temperature=0.4
        )
        models_used["summary"] = summary_result["provider"]
        summary = summary_result["response"]

        # 3. Assess Risk
        risk_result = self.llm_service.generate(
            prompt=f"Assess the risks in the following legal document: {text[:8000]}",
            capability=LLMCapability.ANALYSIS,
            system_prompt=LEGAL_SYSTEM_PROMPTS["risk_assessment"],
            temperature=0.5
        )
        models_used["risk_assessment"] = risk_result["provider"]
        risk_assessment = risk_result["response"]

        # 4. Extract Clauses and Parties using rules/spaCy
        clauses = self._extract_clauses(text)
        parties = self._extract_parties(text)

        end_time = time.time()

        return {
            "document_type": doc_type,
            "summary": summary,
            "risk_assessment": risk_assessment,
            "clauses": clauses,
            "parties": parties,
            "models_used": models_used,
            "processing_time": end_time - start_time,
            "full_text": text # Return full text for RAG indexing
        }

    def _extract_text(self, file_path: str) -> str:
        text = ""
        try:
            if file_path.lower().endswith('.pdf'):
                with open(file_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    for page in reader.pages:
                        text += page.extract_text() or ""
            elif file_path.lower().endswith('.docx'):
                doc = docx.Document(file_path)
                for para in doc.paragraphs:
                    text += para.text + '\n'
            else:
                 with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
        return text

    def _extract_clauses(self, text: str) -> List[Dict[str, str]]:
        clauses = []
        # Regex to find clauses (e.g., "1. Introduction", "Article II: Definitions")
        clause_pattern = re.compile(r'^(?:\n|\A)(?:(?:ARTICLE|SECTION|PART)\s+[IVXLC\d]+[.\s-]*|\d+\.\s+)([A-Z][\w\s]+)(?=\n)', re.IGNORECASE | re.MULTILINE)
        matches = list(clause_pattern.finditer(text))
        for i, match in enumerate(matches):
            start = match.end(0)
            end = matches[i+1].start(0) if i + 1 < len(matches) else len(text)
            clauses.append({
                'title': match.group(1).strip(),
                'content': text[start:end].strip()
            })
        return clauses

    def _extract_parties(self, text: str) -> List[Dict[str, str]]:
        parties = []
        # Use spaCy for Named Entity Recognition
        doc = nlp(text[:5000]) # Limit to first 5000 chars for performance
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG"]:
                party_name = ent.text.strip()
                if len(party_name) > 3 and party_name.lower() not in ["agreement", "contract"]:
                    party_type = "individual" if ent.label_ == "PERSON" else "company"
                    # Avoid duplicates
                    if not any(p['name'] == party_name for p in parties):
                        parties.append({
                            'name': party_name,
                            'party_type': party_type
                        })
        return parties
