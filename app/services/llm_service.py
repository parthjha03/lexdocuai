"""
LLM Service for LexDocuAI

This module provides a service for interacting with multiple LLM providers
(Groq, Google Gemini, and DeepInfra Phi-4) with strategies for optimal usage.
"""

import os
import requests
import json
import time
import random
import logging
import threading
from enum import Enum
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
import google.generativeai as genai
from collections import deque
from flask import current_app

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    GROQ = "groq"
    GEMINI = "gemini"
    PHI = "phi"

class LLMCapability(Enum):
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    ANALYSIS = "analysis"

class RotationStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    COST_OPTIMIZED = "cost_optimized"

class LLMClient(ABC):
    def __init__(self):
        self.last_call_time = 0
        self.rate_limit_delay = 1.0
        self.usage_count = 0
        self.error_count = 0

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        pass

    def enforce_rate_limit(self):
        current_time = time.time()
        elapsed = current_time - self.last_call_time
        if elapsed < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - elapsed
            time.sleep(sleep_time)
        self.last_call_time = time.time()
        self.usage_count += 1

class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.rate_limit_delay = 0.5
        self.capabilities = { LLMCapability.SUMMARIZATION: 0.9, LLMCapability.EXTRACTION: 0.9, LLMCapability.CLASSIFICATION: 0.85, LLMCapability.ANALYSIS: 0.95 }
        self.cost_per_1k_tokens = 0.0007

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        self.enforce_rate_limit()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"messages": messages, "model": self.model, "temperature": temperature, "max_tokens": max_tokens}
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"Groq API error: {e}")
            self.error_count += 1
            return "Error: Could not get response from Groq."

class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gemini-1.5-flash-latest"):
        super().__init__()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.rate_limit_delay = 2.0
        self.capabilities = { LLMCapability.SUMMARIZATION: 0.95, LLMCapability.EXTRACTION: 0.8, LLMCapability.CLASSIFICATION: 0.9, LLMCapability.ANALYSIS: 0.9 }
        self.cost_per_1k_tokens = 0.0015

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        self.enforce_rate_limit()
        try:
            # Note: Gemini API uses a different way to set system prompts
            response = self.model.generate_content(f"{system_prompt}\n\n{prompt}" if system_prompt else prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            self.error_count += 1
            return "Error: Could not get response from Gemini."

class PhiClient(LLMClient):
    def __init__(self, api_key: str, model: str = "microsoft/Phi-3-mini-4k-instruct"):
        super().__init__()
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepinfra.com/v1/openai"
        self.rate_limit_delay = 1.5
        self.capabilities = { LLMCapability.SUMMARIZATION: 0.8, LLMCapability.EXTRACTION: 0.85, LLMCapability.CLASSIFICATION: 0.8, LLMCapability.ANALYSIS: 0.8 }
        self.cost_per_1k_tokens = 0.0005

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        self.enforce_rate_limit()
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        messages = []
        if system_prompt: messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload = {"messages": messages, "model": self.model, "temperature": temperature, "max_tokens": max_tokens}
        try:
            response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logger.error(f"DeepInfra (Phi) API error: {e}")
            self.error_count += 1
            return "Error: Could not get response from Phi."

class MultiLLMService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        with self._lock:
            if not hasattr(self, 'initialized'):
                # Configuration is now fetched from the application context
                config = current_app.config
                self.groq_client = GroqClient(api_key=config['GROQ_API_KEY'])
                self.gemini_client = GeminiClient(api_key=config['GOOGLE_API_KEY'])
                self.phi_client = PhiClient(api_key=config['DEEPINFRA_API_KEY'])
                self.clients = { LLMProvider.GROQ: self.groq_client, LLMProvider.GEMINI: self.gemini_client, LLMProvider.PHI: self.phi_client }
                self.strategy = RotationStrategy.CAPABILITY_BASED
                self.provider_queue = deque(self.clients.keys())
                self.total_calls = 0
                self.provider_history = deque(maxlen=50)
                self.initialized = True

    def select_provider(self, capability: LLMCapability) -> LLMProvider:
        if self.strategy == RotationStrategy.ROUND_ROBIN:
            provider = self.provider_queue[0]
            self.provider_queue.rotate(-1)
            return provider
        elif self.strategy == RotationStrategy.CAPABILITY_BASED:
            best_provider = max(self.clients.items(), key=lambda item: item[1].capabilities.get(capability, 0.0))
            return best_provider[0]
        elif self.strategy == RotationStrategy.COST_OPTIMIZED:
            cheapest_provider = min(self.clients.items(), key=lambda item: item[1].cost_per_1k_tokens)
            return cheapest_provider[0]
        else: # Default to round robin
            return self.select_provider(LLMCapability.ANALYSIS)

    def generate(self, prompt: str, capability: LLMCapability, system_prompt: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1000) -> Dict[str, Any]:
        start_time = time.time()
        provider = self.select_provider(capability)
        client = self.clients[provider]
        response = client.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature, max_tokens=max_tokens)
        elapsed_time = time.time() - start_time
        self.total_calls += 1
        self.provider_history.append(provider)
        logger.info(f"Generated response using {provider.value} in {elapsed_time:.2f}s")
        return {"provider": provider.value, "response": response, "processing_time": elapsed_time}

    def set_strategy(self, strategy: RotationStrategy):
        self.strategy = strategy
        logger.info(f"Changed LLM rotation strategy to {strategy.value}")

# Singleton accessor
def get_llm_service() -> MultiLLMService:
    return MultiLLMService()

LEGAL_SYSTEM_PROMPTS = {
    "document_summary": "You are a legal assistant. Provide a concise summary of the key points, parties, dates, and legal implications.",
    "risk_assessment": "You are a legal expert. Assess the risks in the provided document. Be concise.",
    "legal_validation": "You are a document classifier. Determine if the following text is from a legal document. Respond with only 'yes' or 'no'.",
    "document_classification": "You are a legal document classifier. Determine the type of this document (e.g., contract, legislation, case law) and explain your reasoning."
}
