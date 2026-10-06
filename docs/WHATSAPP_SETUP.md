# Running the WhatsApp channel for real (Twilio sandbox + a tunnel)

Time: about 25 minutes the first time. Cost: free (Twilio trial sandbox, free tunnel), plus a few cents of model use.
What you need: a phone with WhatsApp, an email address.

**Why a tunnel:** Twilio is on the internet and calls your webhook; your API runs on your laptop. A tunnel gives your
laptop a temporary public https address. **Safety:** run with `WEBHOOK_ONLY=true` so that, through the tunnel, nothing
but `/health` and `/channels/whatsapp/*` can be reached (login, documents and the admin API answer 404).

## A. Twilio account and sandbox (one time)
1. Create a free account at twilio.com (email + phone verification). A trial account is enough.
2. Console > **Account info**: copy the **Auth Token** (click the eye icon). It goes into `.env` as `TWILIO_AUTH_TOKEN`.
3. Console > **Messaging > Try it out > Send a WhatsApp message**. You will see the sandbox number
   (`+1 415 523 8886`) and a join phrase such as `join bright-tiger`.
4. On your phone, open WhatsApp and send that exact join phrase to the sandbox number. Twilio replies "You are all set".
   (A phone has to do this once; the sandbox expires after 72 hours of silence.)

## B. A tunnel (each session)
1. Install `cloudflared` (free, no account needed for a quick tunnel): see Cloudflare's download page, or
   `curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared`.
2. `cloudflared tunnel --url http://localhost:8001` prints an address like `https://random-words.trycloudflare.com`. It changes every run.

## C. Configure and start
1. `.env`: set `TWILIO_AUTH_TOKEN=...`, `PUBLIC_BASE_URL=https://random-words.trycloudflare.com` (no trailing slash),
   `WEBHOOK_ONLY=true`, and `ENVIRONMENT=prod`.
2. Make sure a company assistant and public information exist:
   - `uv run python -m scripts.create_whatsapp_channel --site-key <pk_...> --keyword initech`
   - `uv run python -m scripts.set_visibility --org Initech --source initech_handbook --visibility public`
3. `uv run uvicorn app.api.main:_default_app --factory --port 8001`
4. `uv run python -m scripts.whatsapp_preflight` must print 0 failed. It prints the exact webhook address to paste next.

## D. Connect Twilio
Console > Messaging > Try it out > Send a WhatsApp message > **Sandbox settings**:
- *When a message comes in*: `https://random-words.trycloudflare.com/channels/whatsapp/webhook`, method **POST**. Save.

## E. Try it (from your phone, to the sandbox number)
| You send | You should get |
|---|---|
| `hello` | "To start, send: START <company code>" |
| `START initech` | a welcome that says it is an AI assistant, plus HELP |
| `How many vacation days does Initech give?` | an answer with a `Sources:` line |
| `What is Ann's salary band?` | a refusal with "Reply HUMAN..." (it is private) |
| `human`, then `yes <your name>` | consent question, then "A person will contact you" |
| `STOP` | "Everything we held about this chat has been deleted" |

Then in the Streamlit admin > "Leads & handoffs", the handoff and the lead (name, number) appear.

## F. If something goes wrong
- **No reply at all:** Twilio console > Monitor > Logs > Messaging shows the webhook's HTTP status.
- **403:** the signature did not match. Almost always `PUBLIC_BASE_URL` differs from what Twilio calls (old tunnel address, `http` vs `https`, a trailing slash) or the Auth Token is wrong. Restart the API after changing `.env`.
- **404:** the request path is not `/channels/whatsapp/webhook`.
- **"Sorry, something went wrong":** look at the API's log (the error is logged there, never shown to the user).
- **Replies in English only / refusals:** the assistant answers only from PUBLIC documents; check step C2.

## G. Clean up
Stop `cloudflared` (the address dies at once), remove `PUBLIC_BASE_URL` and `WEBHOOK_ONLY` from `.env`, and make the
demo document private again: `... scripts.set_visibility --org Initech --source initech_handbook --visibility org`.
