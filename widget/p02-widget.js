/*
 * P02 chat widget: ONE script tag puts an AI assistant on any website.
 *
 *   <script src="https://API_HOST/widget.js" data-site-key="pk_..." defer></script>
 *
 * Optional attributes: data-api (API address; default = where this script came from),
 *                      data-title, data-color (button/header colour).
 *
 * How it works: a floating button opens a chat panel inside a Shadow DOM (so the host page's CSS
 * cannot break it and ours cannot leak out). Each question is POSTed to /widget/<site key>/chat
 * and the answer is read as a stream of Server-Sent Events. The visitor is anonymous: the server
 * answers only from the company's PUBLIC documents. Text from the server is only ever put on the
 * page with textContent (never as HTML), so a malicious answer cannot inject markup or scripts.
 */
(function () {
  "use strict";

  // ---- pure helper: split a chunk of Server-Sent-Events text into events (tested with Node) ----
  // "event: token\ndata: {\"text\":\"Hi\"}\n\nevent: do" -> events [{event:"token",data:{text:"Hi"}}], rest "event: do"
  function parseSSEBuffer(buffer) {
    var events = [];
    var parts = buffer.split("\n\n");
    var rest = parts.pop(); // the last piece may be an unfinished event
    parts.forEach(function (block) {
      var name = null, data = null;
      block.split("\n").forEach(function (line) {
        if (line.indexOf("event: ") === 0) name = line.slice(7).trim();
        else if (line.indexOf("data: ") === 0) data = line.slice(6);
      });
      if (name && data) {
        try { events.push({ event: name, data: JSON.parse(data) }); } catch (e) { /* skip a broken event */ }
      }
    });
    return { events: events, rest: rest };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { parseSSEBuffer: parseSSEBuffer };
    return; // loaded by a test, not by a browser page
  }

  var script = document.currentScript;
  if (!script) return;
  var siteKey = script.getAttribute("data-site-key");
  if (!siteKey) { console.error("p02-widget: data-site-key is missing"); return; }
  var api = (script.getAttribute("data-api") || new URL(script.src).origin).replace(/\/$/, "");
  var color = script.getAttribute("data-color") || "#2b5fd9";
  var title = script.getAttribute("data-title") || "Chat with us";
  var base = api + "/widget/" + encodeURIComponent(siteKey);
  var sessionKey = "p02-session-" + siteKey;

  function store(key, value) {
    try { if (value === undefined) return sessionStorage.getItem(key); sessionStorage.setItem(key, value); } catch (e) { return null; }
  }
  var sessionId = store(sessionKey);

  // ---- tiny DOM helper ------------------------------------------------------------------------
  function el(tag, attrs, text) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { node.setAttribute(k, attrs[k]); });
    if (text !== undefined) node.textContent = text;
    return node;
  }

  // ---- the panel ------------------------------------------------------------------------------
  var host = el("div");
  var root = host.attachShadow({ mode: "open" });
  var style = el("style", {}, [
    ":host{all:initial}",
    "*{box-sizing:border-box;font-family:system-ui,-apple-system,Segoe UI,sans-serif}",
    ".btn{position:fixed;right:20px;bottom:20px;width:56px;height:56px;border-radius:50%;border:0;background:" + color + ";color:#fff;font-size:26px;cursor:pointer;box-shadow:0 4px 14px rgba(0,0,0,.3);z-index:2147483646}",
    ".panel{position:fixed;right:20px;bottom:88px;width:360px;max-width:calc(100vw - 24px);height:520px;max-height:calc(100vh - 110px);background:#fff;color:#1a1a1a;border-radius:12px;box-shadow:0 8px 30px rgba(0,0,0,.3);display:none;flex-direction:column;overflow:hidden;z-index:2147483646}",
    ".panel.open{display:flex}",
    ".head{background:" + color + ";color:#fff;padding:12px 14px;font-weight:600;display:flex;justify-content:space-between;align-items:center}",
    ".head button{background:none;border:0;color:#fff;font-size:20px;cursor:pointer}",
    ".note{font-size:11.5px;color:#555;background:#f3f5f9;padding:8px 12px;border-bottom:1px solid #e5e8ee}",
    ".msgs{flex:1;overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:10px}",
    ".m{max-width:88%;padding:8px 11px;border-radius:10px;white-space:pre-wrap;word-wrap:break-word;font-size:14px;line-height:1.4}",
    ".m.user{align-self:flex-end;background:" + color + ";color:#fff}",
    ".m.bot{align-self:flex-start;background:#eef1f6}",
    ".src{font-size:11px;color:#555;margin-top:6px}",
    ".tools{display:flex;gap:6px;margin-top:6px}",
    ".tools button,.link{font-size:12px;background:none;border:1px solid #c7ccd6;border-radius:6px;padding:2px 8px;cursor:pointer;color:#333}",
    ".form{display:flex;border-top:1px solid #e5e8ee}",
    ".form input{flex:1;border:0;padding:12px;font-size:14px;outline:none}",
    ".form button{border:0;background:" + color + ";color:#fff;padding:0 16px;cursor:pointer;font-size:14px}",
    ".foot{padding:6px 12px;border-top:1px solid #e5e8ee;text-align:center}",
    ".lead{display:flex;flex-direction:column;gap:6px;background:#f8f9fb;border:1px solid #dde1e9;border-radius:8px;padding:10px}",
    ".lead input,.lead textarea{border:1px solid #c7ccd6;border-radius:6px;padding:6px;font-size:13px}",
    ".lead label{font-size:11.5px;color:#444}",
    ".lead .go{background:" + color + ";color:#fff;border:0;border-radius:6px;padding:7px;cursor:pointer}",
  ].join("\n"));
  var button = el("button", { class: "btn", "aria-label": "Open chat" }, "💬");
  var panel = el("div", { class: "panel", role: "dialog", "aria-label": title });
  var headTitle = el("span", {}, title);
  var closeBtn = el("button", { "aria-label": "Close chat" }, "×");
  var head = el("div", { class: "head" });
  head.append(headTitle, closeBtn);
  var note = el("div", { class: "note" }, "You are chatting with an AI assistant.");
  var msgs = el("div", { class: "msgs", "aria-live": "polite" });
  var input = el("input", { type: "text", maxlength: "500", placeholder: "Type your question…", "aria-label": "Your question" });
  var send = el("button", { type: "submit" }, "Send");
  var form = el("form", { class: "form" });
  form.append(input, send);
  var foot = el("div", { class: "foot" });
  var talk = el("button", { class: "link", type: "button" }, "Talk to a person");
  foot.append(talk);
  panel.append(head, note, msgs, form, foot);
  root.append(style, button, panel);
  document.body.appendChild(host);

  function add(role, text) {
    var m = el("div", { class: "m " + role }, text);
    msgs.appendChild(m);
    msgs.scrollTop = msgs.scrollHeight;
    return m;
  }

  function post(path, body) {
    return fetch(base + path, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
  }

  async function explain(res) {
    if (res.status === 429) return "Too many messages, please wait a moment and try again.";
    try { var j = await res.json(); if (j.detail) return String(j.detail); } catch (e) { /* not JSON */ }
    return "Something went wrong, please try again.";
  }

  // ---- thumbs, sources, handoff, lead form ------------------------------------------------------
  function rate(messageId, rating, holder) {
    post("/feedback", { session_id: sessionId, message_id: messageId, rating: rating })
      .then(function () { holder.textContent = "Thanks for the feedback."; })
      .catch(function () { /* feedback is optional: stay quiet */ });
  }

  function decorate(bubble, done) {
    if (done.citations && done.citations.length) {
      var names = done.citations.map(function (c) { return c.section || c.document; });
      bubble.appendChild(el("div", { class: "src" }, "Sources: " + Array.from(new Set(names)).join(" · ")));
    }
    if (done.message_id && !done.refused) {
      var tools = el("div", { class: "tools" });
      var up = el("button", { type: "button", "aria-label": "Helpful" }, "👍");
      var down = el("button", { type: "button", "aria-label": "Not helpful" }, "👎");
      up.onclick = function () { rate(done.message_id, 1, tools); };
      down.onclick = function () { rate(done.message_id, -1, tools); };
      tools.append(up, down);
      bubble.appendChild(tools);
    }
    if (done.suggest_handoff) {
      var ask = el("button", { class: "link", type: "button" }, "Talk to a person");
      ask.onclick = requestPerson;
      bubble.appendChild(el("div", { class: "tools" })).appendChild(ask);
    }
  }

  function leadForm() {
    var box = el("div", { class: "lead" });
    var name = el("input", { placeholder: "Your name", maxlength: "120", "aria-label": "Your name" });
    var email = el("input", { placeholder: "Email", type: "email", maxlength: "254", "aria-label": "Email" });
    var phone = el("input", { placeholder: "Phone (optional)", maxlength: "40", "aria-label": "Phone" });
    var message = el("textarea", { placeholder: "How can we help? (optional)", rows: "2", maxlength: "1000", "aria-label": "Message" });
    var consent = el("input", { type: "checkbox", id: "consent" });
    var label = el("label", { for: "consent" }, " I agree that this company may contact me about my request using these details.");
    var go = el("button", { class: "go", type: "button" }, "Send my details");
    var status = el("div", { class: "src" });
    go.onclick = async function () {
      var res = await post("/lead", {
        session_id: sessionId, name: name.value, email: email.value, phone: phone.value || null,
        message: message.value || null, consent: consent.checked,
      }).catch(function () { return null; });
      if (res && res.ok) { box.replaceChildren(el("div", {}, "Thank you. We have your details and will be in touch.")); }
      else { status.textContent = res ? await explain(res) : "Could not reach the server."; }
    };
    box.append(name, email, phone, message, consent, label, go, status);
    msgs.appendChild(box);
    msgs.scrollTop = msgs.scrollHeight;
  }

  async function requestPerson() {
    if (!sessionId) { add("bot", "Ask a question first, then I can pass you on to a person."); return; }
    var res = await post("/handoff", { session_id: sessionId }).catch(function () { return null; });
    if (res && res.ok) { add("bot", (await res.json()).message); leadForm(); }
    else add("bot", res ? await explain(res) : "Could not reach the server.");
  }

  // ---- asking a question ------------------------------------------------------------------------
  var busy = false;
  async function ask(question) {
    busy = true; send.disabled = true;
    add("user", question);
    var bubble = add("bot", "…");
    try {
      var res = await post("/chat", { question: question, session_id: sessionId });
      if (!res.ok) {
        bubble.textContent = await explain(res);
        if (res.status === 429) leadForm();
        return;
      }
      var reader = res.body.getReader(), decoder = new TextDecoder(), buffer = "", shown = "";
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var parsed = parseSSEBuffer(buffer);
        buffer = parsed.rest;
        parsed.events.forEach(function (ev) {
          if (ev.event === "token") { shown += ev.data.text; bubble.textContent = shown; msgs.scrollTop = msgs.scrollHeight; }
          else if (ev.event === "done") {
            sessionId = ev.data.session_id; store(sessionKey, sessionId);
            bubble.textContent = ev.data.text;  // the validated answer replaces what streamed
            decorate(bubble, ev.data);
          } else if (ev.event === "error") { bubble.textContent = ev.data.message; }
        });
      }
    } catch (e) {
      bubble.textContent = "Could not reach the server. Please try again.";
    } finally {
      busy = false; send.disabled = false; input.focus();
    }
  }

  form.onsubmit = function (e) {
    e.preventDefault();
    var q = input.value.trim();
    if (!q || busy) return;
    input.value = "";
    ask(q);
  };
  talk.onclick = requestPerson;
  button.onclick = closeBtn.onclick = function () {
    panel.classList.toggle("open");
    if (panel.classList.contains("open")) input.focus();
  };

  // greet, and use the site's own display name and disclosure line from the server
  fetch(base + "/config").then(function (r) { return r.ok ? r.json() : null; }).then(function (c) {
    if (!c) return;
    if (!script.getAttribute("data-title") && c.name) headTitle.textContent = c.name;
    if (c.disclosure) note.textContent = c.disclosure;
  }).catch(function () { /* defaults are fine */ });
  add("bot", "Hi! Ask me anything about us.");
})();
