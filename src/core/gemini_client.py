"""
Gemini client factory.
Returns a configured google.genai.Client ready for async use.
"""
from __future__ import annotations
import os
from google import genai


def get_gemini_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY", "")
    return genai.Client(api_key=api_key)
