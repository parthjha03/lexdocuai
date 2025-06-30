"""
RAG and Chat Service for LexDocuAI
"""

import logging
import os
from typing import List, Dict, Any
from flask import current_app

from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI # For DeepInfra endpoint
from groq import Groq
from langdetect import detect
from deep_translator import GoogleTranslator
import spacy
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import CrossEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingWrapper:
    """A wrapper for the DeepInfra embedding API."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("DeepInfra API key is required.")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
        self.model = "BAAI/bge-base-en-v1.5"

    def create_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embeddings.create(model=self.model, input=batch, encoding_format="float")
                all_embeddings.extend([res.embedding for res in response.data])
            except Exception as e:
                logger.error(f"Error creating embeddings: {e}")
                all_embeddings.extend([[]] * len(batch))
        return all_embeddings

class RAGService:
    def __init__(self):
        config = current_app.config
        self.config = config
        self.embedding_wrapper = EmbeddingWrapper(api_key=config['DEEPINFRA_API_KEY'])
        self.groq_client = Groq(api_key=config['GROQ_API_KEY'])
        
        pc = Pinecone(api_key=config['PINECONE_API_KEY'])
        index_name = config['PINECONE_INDEX_NAME']
        required_metric = 'dotproduct'  # For sparse-dense vectors

        # Check if the index exists
        if index_name in pc.list_indexes().names():
            index_description = pc.describe_index(index_name)
            # If the metric is not 'dotproduct', delete and recreate the index
            if index_description.metric != required_metric:
                logger.warning(
                    f"Index '{index_name}' found with incorrect metric ('{index_description.metric}'). "
                    f"Required metric is '{required_metric}'. Deleting and recreating index. "
                    f"All existing data will be lost."
                )
                pc.delete_index(index_name)
                pc.create_index(
                    name=index_name,
                    dimension=config['EMBEDDING_DIMENSION'],
                    metric=required_metric,
                    spec=ServerlessSpec(cloud=config['PINECONE_CLOUD'], region=config['PINECONE_REGION'])
                )
        # If the index does not exist, create it
        else:
            logger.info(f"Index '{index_name}' not found. Creating new index with '{required_metric}' metric.")
            pc.create_index(
                name=index_name,
                dimension=config['EMBEDDING_DIMENSION'],
                metric=required_metric,
                spec=ServerlessSpec(cloud=config['PINECONE_CLOUD'], region=config['PINECONE_REGION'])
            )

        self.pinecone_index = pc.Index(index_name)
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize models and components for advanced RAG
        self.vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_df=0.85, min_df=2)
        self.vectorizer_fitted = False
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Conversation histories
        self.conversation_history = []
        self.rewrite_conversation_history = []

    def delete_document(self, document_id: int):
        """Deletes all vectors associated with a document_id from Pinecone."""
        try:
            logger.info(f"Attempting to delete all vectors for document_id: {document_id}")
            # A dummy vector is used as the query itself is not important, only the filter
            dummy_vector = [0.0] * self.config['EMBEDDING_DIMENSION']
            query_response = self.pinecone_index.query(
                vector=dummy_vector,
                filter={'document_id': document_id},
                top_k=10000,  # Fetch up to 10,000 vector IDs to be safe
                include_values=False,
                include_metadata=False
            )
            ids_to_delete = [match['id'] for match in query_response.get('matches', [])]

            if ids_to_delete:
                self.pinecone_index.delete(ids=ids_to_delete)
                logger.info(f"Successfully deleted {len(ids_to_delete)} vectors for document_id: {document_id}")
            else:
                logger.info(f"No vectors found for document_id: {document_id}. Nothing to delete.")
        except Exception as e:
            logger.error(f"An error occurred while deleting vectors for document {document_id}: {e}")

    def index_document(self, document_id: int, text: str):
        logger.info(f"Starting to index document_id: {document_id}")

        # Delete existing vectors for the document before re-indexing
        self.delete_document(document_id)

        chunks = self._semantic_chunking(text)
        if not chunks:
            logger.warning(f"No chunks were generated for document {document_id}")
            return

        # Fit the TF-IDF vectorizer on the first document's chunks
        if not self.vectorizer_fitted:
            logger.info("Fitting TF-IDF vectorizer on the first document...")
            self.vectorizer.fit(chunks)
            self.vectorizer_fitted = True
        
        embeddings = self.embedding_wrapper.create_embeddings(chunks)
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:
                # Create sparse vector
                sparse_vec = self.vectorizer.transform([chunk])
                sparse_values = {
                    'indices': sparse_vec.indices.tolist(),
                    'values': sparse_vec.data.tolist()
                }

                vectors.append({
                    'id': f'doc_{document_id}_chunk_{i}',
                    'values': embedding,
                    'sparse_values': sparse_values,
                    'metadata': {'document_id': document_id, 'text': chunk}
                })

        if vectors:
            self.pinecone_index.upsert(vectors=vectors)
            logger.info(f"Successfully indexed {len(vectors)} chunks for document_id: {document_id}")
        else:
            logger.error(f"No vectors were generated for document {document_id}")

    def chat_with_document(self, document_id: int, user_question: str) -> str:
        # 1. Translate and rewrite query
        original_lang = detect(user_question)
        english_question = self._translate_text(user_question, 'en', original_lang)
        rewritten_query = self._rewrite_query(english_question)
        logger.info(f"Rewritten query for search: '{rewritten_query}'")

        # 2. Hybrid search to retrieve initial candidates
        logger.info("Performing hybrid search for candidate chunks...")
        candidate_matches = self._hybrid_search(document_id, rewritten_query, top_k=20)

        # 3. Re-rank candidates with Cross-Encoder
        logger.info(f"Re-ranking {len(candidate_matches)} candidates with Cross-Encoder...")
        reranked_matches = self._rerank_with_cross_encoder(rewritten_query, candidate_matches)
        
        # 4. Select top N chunks for context
        top_k_reranked = 5
        relevant_chunks = [match['metadata']['text'] for match in reranked_matches[:top_k_reranked]]
        logger.info(f"Selected top {len(relevant_chunks)} re-ranked chunks for context.")

        # 5. Generate response
        response_en = self._get_chatbot_response(english_question, relevant_chunks)

        # 6. Translate back and return
        final_response = self._translate_text(response_en, original_lang, 'en')
        return final_response

    def _semantic_chunking(self, text: str, max_chunk_length: int = 500) -> List[str]:
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < max_chunk_length:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def _hybrid_search(self, document_id: int, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.vectorizer_fitted:
            logger.warning("Vectorizer not fitted. Cannot perform sparse search.")
            return []

        # Create dense and sparse vectors for the query
        dense_vec = self.embedding_wrapper.create_embeddings([query])[0]
        sparse_vec = self.vectorizer.transform([query])
        sparse_values = {
            'indices': sparse_vec.indices.tolist(),
            'values': sparse_vec.data.tolist()
        }

        if not dense_vec:
            return []
        
        results = self.pinecone_index.query(
            vector=dense_vec,
            sparse_vector=sparse_values,
            filter={'document_id': document_id},
            top_k=top_k,
            include_metadata=True
        )
        return results.get('matches', [])

    def _rerank_with_cross_encoder(self, query: str, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not matches:
            return []

        # Create pairs of [query, chunk_text] for the cross-encoder
        # The match object from pinecone-client v4 acts like a dict but is a model.
        # Accessing metadata via attribute is safer.
        pairs = [(query, match.metadata['text']) for match in matches]

        # Predict scores
        scores = self.cross_encoder.predict(pairs)

        # Combine matches with their new scores into a new list of plain dicts
        # to avoid modifying the original Pinecone model objects.
        reranked_matches = []
        for match, score in zip(matches, scores):
            reranked_matches.append({
                'id': match.id,
                'metadata': match.metadata,
                'rerank_score': float(score)  # Ensure score is a standard float
            })

        # Sort the results by the new rerank_score
        return sorted(reranked_matches, key=lambda x: x['rerank_score'], reverse=True)

    def _rewrite_query(self, query: str) -> str:
        system_prompt = '''
    You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system. The RAG system has information about various documents.
    Given the original query and the conversation history, rewrite it to be more specific, detailed, and likely to retrieve relevant information. Do not make up information that is not in the question, although you are free to add details if they were mentioned earlier in the conversation history.
    Consider the context of the conversation when rewriting the query. You are rewriting the queries such that they can be used for semantic search in a RAG system whose information will be passed on to another LLM for response. Keep this in mind. Not every query needs rewriting; use your judgment on when to rewrite and when not to. ONLY give the rewritten query as output.
    '''
        messages = [
            {"role": "system", "content": system_prompt},
            *self.rewrite_conversation_history,
            {"role": "user", "content": f"Original query: {query}\n\nRewritten query:"}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                max_tokens=200,
                temperature=0.3
            )
            rewritten_query = response.choices[0].message.content.strip()
            logger.info(f"Query rewritten to: '{rewritten_query}'")

            # Update and manage rewrite history
            self.rewrite_conversation_history.append({"role": "user", "content": query})
            self.rewrite_conversation_history.append({"role": "assistant", "content": rewritten_query})
            if len(self.rewrite_conversation_history) > 10: # Keep last 5 turns
                self.rewrite_conversation_history = self.rewrite_conversation_history[-10:]
            
            return rewritten_query
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return query

    def _get_chatbot_response(self, user_question: str, relevant_chunks: List[str]) -> str:
        system_prompt = '''
    You are an AI assistant tasked with generating responses based on user questions and relevant information chunks. 
    The information chunks are related to various documents. Use the provided chunks to generate a detailed and accurate response to the user's question. 
    Ensure that the response is relevant and informative, and avoid making up information that is not present in the chunks.
    '''
        context = "\n\n---\n\n".join(relevant_chunks)

        messages = [
            {"role": "system", "content": system_prompt},
            *self.conversation_history,
            {"role": "user", "content": f"User question: {user_question}\n\nRelevant chunks: {context}\n\nResponse:"}
        ]

        try:
            response = self.groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=messages,
                max_tokens=4000,
                temperature=0.3
            )
            chatbot_response = response.choices[0].message.content.strip()
            
            # Update and manage conversation history
            self.conversation_history.append({"role": "user", "content": user_question})
            self.conversation_history.append({"role": "assistant", "content": chatbot_response})
            if len(self.conversation_history) > 10: # Keep last 5 turns
                self.conversation_history = self.conversation_history[-10:]

            return chatbot_response
        except Exception as e:
            logger.error(f"Error getting chatbot response: {e}")
            return "I'm sorry, I encountered an error while generating a response."

    def _translate_text(self, text: str, target_lang: str, source_lang: str) -> str:
        if source_lang == target_lang:
            return text
        try:
            return GoogleTranslator(source=source_lang, target=target_lang).translate(text)
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text # Return original text if translation fails

_rag_service_instance = None

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
