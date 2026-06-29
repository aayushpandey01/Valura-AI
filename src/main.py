"""
Valura AI — FastAPI microservice entry point.
Uses Gemini instead of OpenAI. All LLM calls go through google-genai.
"""
from __future__ import annotations
import asyncio
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from google import genai

from .core.models import AgentRequest, PipelineMetadata, QueryRequest
from .core.safety import check_safety
from .core.classifier import classify_intent
from .core.session import get_store
from .agents.router import route

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
_PIPELINE_TIMEOUT_SECONDS = int(os.getenv("PIPELINE_TIMEOUT_SECONDS", "30"))

_gemini_client: genai.Client | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _gemini_client
    # Only create a real client if a key is available.
    # In tests, _gemini_client is injected by the fixture before requests are made.
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key:
        _gemini_client = genai.Client(api_key=api_key)
    logger.info("Valura AI started. Model: %s, Timeout: %ds", _GEMINI_MODEL, _PIPELINE_TIMEOUT_SECONDS)
    yield
    logger.info("Valura AI shutting down.")


app = FastAPI(
    title="Valura AI",
    description="AI co-investor microservice — classify, route, stream.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(data: dict | str) -> str:
    if isinstance(data, dict):
        data = json.dumps(data)
    return f"data: {data}\n\n"


async def _run_pipeline(body: QueryRequest, conversation_id: str) -> AsyncIterator[str]:
    store = get_store()
    client = _gemini_client

    # 1. Safety guard — sync, no LLM
    safety_result = check_safety(body.query)
    if safety_result.blocked:
        logger.info("Safety blocked: conv=%s category=%s", conversation_id, safety_result.category)
        yield _sse({"type": "blocked", "category": safety_result.category.value, "message": safety_result.response})
        yield _sse("[DONE]")
        return

    # 2. Load session history
    history = store.get_history(conversation_id)

    # 3. Classify intent
    try:
        classifier_output = await asyncio.wait_for(
            classify_intent(body.query, history, client, _GEMINI_MODEL),
            timeout=8.0,
        )
    except asyncio.TimeoutError:
        yield _sse({"type": "error", "code": "CLASSIFIER_TIMEOUT", "message": "Request timed out during classification."})
        yield _sse("[DONE]")
        return

    # 4. Emit metadata
    meta = PipelineMetadata(
        agent=classifier_output.agent,
        intent=classifier_output.intent,
        entities=classifier_output.entities,
        safety_verdict=classifier_output.safety_verdict,
        classifier_confidence=classifier_output.confidence,
        safety_latency_ms=safety_result.latency_ms,
        conversation_id=conversation_id,
    )
    yield _sse({"type": "metadata", **meta.model_dump()})

    # 5. Store user turn
    store.add_turn(conversation_id, "user", body.query)

    # 6. Build agent request
    agent_request = AgentRequest(
        query=body.query,
        user_context=body.user_context,
        classifier_output=classifier_output,
        conversation_history=history,
    )

    # 7. Route and stream
    agent_chunks: list[str] = []
    try:
        async with asyncio.timeout(_PIPELINE_TIMEOUT_SECONDS - 8):
            async for chunk in route(agent_request, client, _GEMINI_MODEL):
                agent_chunks.append(chunk)
                yield chunk
    except asyncio.TimeoutError:
        yield _sse({"type": "error", "code": "AGENT_TIMEOUT", "message": "Agent response timed out."})
        yield _sse("[DONE]")
        return
    except Exception as e:
        logger.exception("Agent error: conv=%s agent=%s", conversation_id, classifier_output.agent)
        yield _sse({"type": "error", "code": "AGENT_ERROR", "message": "An unexpected error occurred."})
        yield _sse("[DONE]")
        return

    # 8. Store assistant turn
    store.add_turn(conversation_id, "assistant", "".join(agent_chunks))


@app.post("/query")
async def query(body: QueryRequest):
    conversation_id = body.conversation_id or body.session_id or str(uuid.uuid4())
    return StreamingResponse(
        _run_pipeline(body, conversation_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": _GEMINI_MODEL, "active_sessions": get_store().active_sessions}
