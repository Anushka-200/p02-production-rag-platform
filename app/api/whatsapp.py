"""The WhatsApp webhook: Twilio calls this URL for every message a user sends to our number.

Order of checks (cheapest and most important first):
  1. the signature header proves the request came from Twilio (see app/channels/twilio.py)
  2. the conversation logic runs (app/channels/whatsapp.py) and returns the reply text
  3. the reply goes back as TwiML (XML), which Twilio delivers to the user
Any internal error becomes a short apology, never a stack trace, because the reply is shown to a
stranger on their phone.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.channels.twilio import empty_twiml, signature_is_valid, twiml_reply
from app.channels.whatsapp import Deps, handle_message

logger = logging.getLogger(__name__)
SORRY = "Sorry, something went wrong on our side. Please try again in a moment."


def webhook_url(request: Request, public_base_url: str | None) -> str:
    """The URL Twilio called. Behind a tunnel or proxy our own view of it differs, so the public
    address can be configured; the signature is computed over exactly what Twilio called."""
    if public_base_url:
        query = f"?{request.url.query}" if request.url.query else ""
        return public_base_url.rstrip("/") + request.url.path + query
    return str(request.url)


def build_whatsapp_router(deps: Deps) -> APIRouter:
    router = APIRouter(prefix="/channels/whatsapp")
    s = deps.settings

    @router.post("/webhook")
    async def webhook(request: Request) -> Response:
        form = await request.form()
        params: dict[str, Any] = {key: str(value) for key, value in form.items()}
        if not s.twilio_verify_signature:
            if s.environment != "dev":  # a switch for local testing, never for a real deployment
                raise HTTPException(503, "signature checking cannot be disabled outside dev")
        else:
            if s.twilio_auth_token is None:
                raise HTTPException(503, "WhatsApp is not configured")
            if not signature_is_valid(
                s.twilio_auth_token.get_secret_value(),
                webhook_url(request, s.public_base_url),
                params,
                request.headers.get("X-Twilio-Signature"),
            ):
                logger.warning("whatsapp webhook with a bad signature refused")
                raise HTTPException(403, "invalid signature")
        try:
            reply = await run_in_threadpool(
                handle_message, deps, params.get("From", ""), params.get("To", ""),
                params.get("Body", ""), params.get("MessageSid", ""),
            )  # fmt: skip
        except Exception:
            logger.exception("whatsapp message failed")
            reply = SORRY
        return Response(
            twiml_reply(reply) if reply else empty_twiml(), media_type="application/xml"
        )

    return router
