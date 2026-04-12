"""Chat endpoints — POST and WebSocket."""

from __future__ import annotations

import asyncio
import json
import uuid
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.shared.schemas import ChatRequest, ChatResponse
from src.gateway.mcp_client import call_orchestrator_tool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and receive a response from the orchestrator pipeline.

    Pass ``stream=true`` in the request body to receive a Server-Sent Events
    stream instead of a single JSON response.
    """
    session_id = request.session_id or str(uuid.uuid4())
    repo_id = request.repo_id or ""
    model = request.model or ""

    result = await call_orchestrator_tool(
        "route_to_agents",
        {"message": request.message, "session_id": session_id, "repo_id": repo_id, "model": model},
        timeout=300,
    )

    if "error" in result and "final_response" not in result:
        if request.stream:
            async def _error_sse():
                yield f"data: {json.dumps({'type': 'error', 'error': result['error']})}\n\n"
                yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return StreamingResponse(_error_sse(), media_type="text/event-stream")
        return ChatResponse(
            response=f"Error: {result['error']}",
            session_id=session_id,
            agents_used=[],
        )

    agents_used = result.get("agent_plan", [])
    final_response = result.get("final_response", str(result))
    agent_results = result.get("agent_results", {})
    tool_plan = result.get("tool_plan", [])

    if request.stream:
        async def _sse():
            yield (
                f"data: {json.dumps({'type': 'metadata', 'session_id': session_id, 'agents_used': agents_used})}\n\n"
            )
            # Emit in 80-char chunks; asyncio.sleep(0) yields control between chunks
            for i in range(0, len(final_response), 80):
                yield f"data: {json.dumps({'type': 'chunk', 'content': final_response[i:i+80]})}\n\n"
                await asyncio.sleep(0)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        return StreamingResponse(_sse(), media_type="text/event-stream")

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        agents_used=agents_used,
        agent_results=agent_results,
        tool_plan=tool_plan,
    )


@router.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    """Real-time chat over WebSocket with streaming responses."""
    await websocket.accept()
    session_id = str(uuid.uuid4())

    try:
        while True:
            data = await websocket.receive_text()

            try:
                payload = json.loads(data)
                message = payload.get("message", data)
                session_id = payload.get("session_id", session_id)
                repo_id = payload.get("repo_id", "")
                model = payload.get("model", "")
            except (json.JSONDecodeError, TypeError):
                message = data
                repo_id = ""
                model = ""

            # Send acknowledgement
            await websocket.send_json(
                {"type": "ack", "session_id": session_id}
            )

            result = await call_orchestrator_tool(
                "route_to_agents",
                {"message": message, "session_id": session_id, "repo_id": repo_id, "model": model},
            )

            agents_used = result.get("agent_plan", [])
            final_response = result.get("final_response", str(result))

            await websocket.send_json(
                {
                    "type": "response",
                    "response": final_response,
                    "session_id": session_id,
                    "agents_used": agents_used,
                }
            )
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected (session %s)", session_id)
    except Exception as exc:
        logger.error("WebSocket error: %s", exc)
        try:
            await websocket.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
