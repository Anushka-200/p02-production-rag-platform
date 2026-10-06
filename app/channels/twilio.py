"""Twilio helpers: verify a webhook really came from Twilio, build the reply, hash phone numbers.

Signature check (what Twilio documents): Twilio takes the full URL it called, appends every POST
field as name+value in alphabetical order, signs that string with HMAC-SHA1 using YOUR auth
token, and sends the base64 result in the `X-Twilio-Signature` header. We repeat the calculation
and compare. Anyone who does not know the auth token cannot produce a matching signature, so a
stranger who finds the webhook URL cannot make the bot talk.

    url    https://x.example/channels/whatsapp/webhook
    params {"Body": "Hi", "From": "whatsapp:+4915112345678"}
    signed https://x.example/channels/whatsapp/webhookBodyHiFromwhatsapp:+4915112345678
"""

import base64
import hashlib
import hmac
from xml.sax.saxutils import escape

XML_HEAD = '<?xml version="1.0" encoding="UTF-8"?>'
MAX_REPLY_CHARS = 1500  # WhatsApp via Twilio accepts at most 1600 characters per message


def compute_signature(auth_token: str, url: str, params: dict[str, str]) -> str:
    payload = url + "".join(f"{key}{params[key]}" for key in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def signature_is_valid(
    auth_token: str, url: str, params: dict[str, str], header: str | None
) -> bool:
    if not header:
        return False
    return hmac.compare_digest(compute_signature(auth_token, url, params), header.strip())


def twiml_reply(text: str) -> str:
    """The XML Twilio expects as the webhook's answer; it sends `text` back to the user."""
    text = text.strip()
    if len(text) > MAX_REPLY_CHARS:
        text = text[: MAX_REPLY_CHARS - 1].rstrip() + "…"
    return f"{XML_HEAD}<Response><Message>{escape(text)}</Message></Response>"


def empty_twiml() -> str:
    return f"{XML_HEAD}<Response></Response>"


def normalise_number(raw: str) -> str:
    """'whatsapp:+49 151 123' -> '+49151123' (one spelling per phone, so one hash per phone)."""
    number = raw.strip().lower().removeprefix("whatsapp:")
    return "+" + "".join(ch for ch in number if ch.isdigit()) if number.startswith("+") else number


def phone_hash(secret: str, number: str) -> str:
    """A keyed hash of the phone number. We store THIS, never the number itself, to recognise a
    returning user. It cannot be reversed without the secret (a plain hash of a phone number could
    be brute-forced: there are only about 10^10 numbers)."""
    key = hmac.new(secret.encode(), b"whatsapp-phone-v1", hashlib.sha256).digest()
    return hmac.new(key, normalise_number(number).encode(), hashlib.sha256).hexdigest()
