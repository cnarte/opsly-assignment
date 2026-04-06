"""Chat endpoints — POST and WebSocket."""

from __future__ import annotations

import json
import uuid
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.shared.schemas import ChatRequest, ChatResponse
from src.gateway.mcp_client import call_orchestrator_tool

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Send a message and receive a response from the orchestrator pipeline."""
    session_id = request.session_id or str(uuid.uuid4())

    result = await call_orchestrator_tool(
        "route_to_agents",
        {"message": request.message, "session_id": session_id},
    )

    if "error" in result and "final_response" not in result:
        return ChatResponse(
            response=f"Error: {result['error']}",
            session_id=session_id,
            agents_used=[],
        )

    agents_used = result.get("agent_plan", [])
    final_response = result.get("final_response", str(result))

    return ChatResponse(
        response=final_response,
        session_id=session_id,
        agents_used=agents_used,
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
            except (json.JSONDecodeError, TypeError):
                message = data

            # Send acknowledgement
            await websocket.send_json(
                {"type": "ack", "session_id": session_id}
            )

            result = await call_orchestrator_tool(
                "route_to_agents",
                {"message": message, "session_id": session_id},
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
