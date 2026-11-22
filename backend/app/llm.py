"""Shared Gemini client. Import from here instead of creating a client per file:

    from app.llm import client, MODEL, types
"""

import warnings

warnings.filterwarnings("ignore", category=SyntaxWarning)  # harmless noise from the SDK

from google import genai
from google.genai import types  # re-exported for convenience

from app.config import GEMINI_API_KEY, GEMINI_MODEL, GEMINI_ROUTER_MODEL

MODEL = GEMINI_MODEL
ROUTER_MODEL = GEMINI_ROUTER_MODEL
client = genai.Client(api_key=GEMINI_API_KEY)

__all__ = ["client", "MODEL", "ROUTER_MODEL", "types"]
