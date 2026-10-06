# ADR 011: Public channels (widget first)

Status: accepted, implemented in Phase 4 block 1 and 3 (2026-10-06)

## Context
Website visitors and WhatsApp users are not org members, cannot log in, and may be hostile. They must get answers from the company's PUBLIC information and nothing else, at a bounded cost, with a way to reach a human.

## Options
1. Treat visitors as a shared "guest" user per org (reuse the existing tables and policies).
2. A separate anonymous scope plus tables keyed by the embedded SITE (chosen).
3. A separate service and database for public chat.

## Decision
- **Anonymous scope** (`Scope(org, user=None, site)`): reads only `public` documents; every write policy requires an owner, so visitors can never write documents. Option 1 was rejected because a shared user would make every visitor's data (chats, leads) visible to each other.
- **A third setting, `app.site_id`**, and policies on the `widget_*`, `leads`, `feedback`, `site_usage` tables that match org AND site. A visitor session key (random UUID) is only a handle for rows of its own site. Staff (members) can read these rows; the API requires an admin.
- **The site key is public** and goes in the URL path (`/widget/{key}/...`) so the browser's CORS preflight, which cannot carry headers, can be answered per site. CORS is dynamic: an origin is allowed only if it is on that site's allow-list; the same check is enforced again on the real request.
- **Rate limit**: fixed window, per site and visitor address, in Redis (falls back to memory with a warning). Behind a proxy `TRUST_PROXY=true` is required to use `X-Forwarded-For`; otherwise a forged header cannot rotate the key.
- **Budget cap**: per-site daily budget in dollars (`site_usage`); the assistant stops with a "leave your details" message when it is reached. It is a soft cap (concurrent requests can overshoot slightly).
- **No long-term memory for visitors.** No personal data is learned about people who never logged in.
- **Lead capture** stores only what the visitor typed, requires an explicit consent tick (also a database CHECK), and the handoff flag is a column on the session that staff resolve.
- The widget never renders server text as HTML (`textContent` only) and is a single file using Shadow DOM.

## Consequences
- Publishing a document publishes ALL of it (a test run showed a public handbook answering a question about its own "lunch code word"); only admins can publish.
- Per-address limits can be bypassed by a botnet; the budget cap is the backstop. CAPTCHA/proof-of-work is a Phase 7 item.
- Visitor chat transcripts are personal data the company now holds; the retention note (Phase 7) must cover `widget_messages` and `leads`.

## WhatsApp via Twilio (Phase 4 block 4, 2026-10-06; migration 0009, `app/channels/`, `app/api/whatsapp.py`)
- **Webhook trust:** every request must carry a valid `X-Twilio-Signature` (HMAC-SHA1 of the called URL plus the sorted form fields, with our auth token). Verified against Twilio's documented example vector (`RSOYDt4T1cUTdK1PDd93/VVr8B8=`). When a tunnel or proxy hides the real URL, `PUBLIC_BASE_URL` is what gets signed. Switching verification off is refused outside `ENVIRONMENT=dev`.
- **Replies are TwiML in the webhook response** (no outbound REST call, no extra credentials; Twilio gives us ~15 s, a cited answer takes 3-5 s). Messages are cut at 1500 characters (limit 1600).
- **Routing without storing phone numbers:** a keyed hash of the number (HMAC-SHA256, key derived from the server secret) is the only identifier kept. A user joins a company with `START <keyword>` (the sandbox has one shared number), or the dedicated number the message was sent to selects the company. The number itself is stored only in `leads`, after the user answers `YES <name>` to "may we contact you here?".
- **Same engine as the web widget:** anonymous scope (public documents only), the same per-site daily budget, the same `widget_*` tables (session `channel = 'whatsapp'`), so staff see handoffs from both channels in one list.
- **Idempotency:** Twilio redelivers when it does not get an answer in time; `whatsapp_seen(message_sid)` makes a duplicate a no-op (no second answer, no second cost).
- **User control:** `HELP`, `HUMAN`, and `STOP`, which deletes the routing row, the conversation, its messages, feedback and any lead.
- **Limits:** 10 messages per minute per phone (hash), then the site's budget cap. Residual risk: a sandbox phone must opt in by joining Twilio's sandbox first, which is Twilio's own gate; a production number needs an approved WhatsApp Business sender (not covered in this project).
