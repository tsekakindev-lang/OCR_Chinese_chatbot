/**
 * app.js
 * Frontend logic for the PDF chat UI:
 * - Upload a PDF
 * - Send chat messages to backend
 * - Render messages in the chat window
 * - Manage UI busy states (spinners/disabled buttons)
 * - (Optionally) clear backend PDF/index
 */

/* ---------------------------------------------------------
 * 1) Endpoint configuration
 * --------------------------------------------------------- */

// Use the same origin (host + port) the page is served from.
// This is useful when deploying so you don’t hardcode URLs.
const BASE = window.location.origin;

// Backend endpoints. These must match your FastAPI routes.
const ENDPOINTS = {
  chat: `${BASE}/chat`,       // POST: send messages, get assistant reply
  upload: `${BASE}/upload`,   // POST: upload PDF
  clear: `${BASE}/clear`      // POST: reset backend state (clear PDF/index)
};

// If your backend is on the same domain, you can also use relative paths:
// const ENDPOINTS = { chat: "/chat", upload: "/upload", clear: "/clear" };

// LocalStorage key used for persisting chat messages (if enabled)
const STORAGE_KEY = "bs_chatbot_messages_v2_sidebar";


/* ---------------------------------------------------------
 * 2) Grab DOM elements
 * --------------------------------------------------------- */

const chatEl = document.getElementById("chat");
const textInput = document.getElementById("textInput");
const sendBtn = document.getElementById("sendBtn");
const clearBtn = document.getElementById("clearBtn");
const cleanPdfBtn = document.getElementById("cleanPdfBtn");

// Upload UI elements
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const filePill = document.getElementById("filePill");

// Spinner elements (Bootstrap spinner)
const uploadSpinner = document.getElementById("uploadSpinner");
const sendSpinner = document.getElementById("sendSpinner");

// Busy flags so we can disable the UI during upload/chat calls
const busy = { uploading: false, thinking: false };


/* ---------------------------------------------------------
 * 3) Message state (load/save/render)
 * --------------------------------------------------------- */

// Load previous messages (if any)
let messages = loadMessages();

// Initial render / reset behavior
// NOTE: Your current behavior ALWAYS clears messages on refresh.
// If you want chat persistence on refresh, remove the reset block below.
messages = [];
try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}

chatEl.innerHTML = "";
filePill.textContent = "No PDF selected";
filePill.title = "";

// Show a welcome message in the chat UI
addMessage("system", "👋 Welcome! Please upload a PDF to begin chatting.");
scrollToBottom();

// Optional: also clear backend state on refresh so it forgets last PDF
fetch(ENDPOINTS.clear, { method: "POST" }).catch(() => {});


/* ---------------------------------------------------------
 * 4) Sidebar actions (Upload / Clear chat / Clear PDF)
 * --------------------------------------------------------- */

// Clicking the Upload button triggers the hidden file input
uploadBtn.addEventListener("click", () => fileInput.click());

/**
 * When the user selects a file:
 * - Validate it's a PDF
 * - Show filename in the UI
 * - Upload to backend (/upload)
 * - Show success/error in chat
 */
fileInput.addEventListener("change", async () => {
  const file = fileInput.files?.[0];
  if (!file) return;

  // Only accept PDFs
  if (file.type !== "application/pdf") {
    addMessage("system", "Please select a PDF file.");
    fileInput.value = "";
    return;
  }

  // Show selected filename
  filePill.textContent = file.name;
  filePill.title = file.name;

  // Add a status message to the chat while uploading
  const uploadStatusId = addMessage("system", `Uploading PDF: ${file.name}`);
  scrollToBottom();

  try {
    // Disable UI + show upload spinner
    setBusy({ uploading: true });

    // Prepare multipart form upload
    const form = new FormData();
    form.append("file", file);

    // Call backend upload endpoint
    const res = await fetch(ENDPOINTS.upload, { method: "POST", body: form });

    // Try to parse JSON response (may fail if backend returns non-JSON)
    let data = {};
    try { data = await res.json(); } catch (_) {}

    // If HTTP not OK, throw a useful error
    if (!res.ok) {
      const msg = data?.detail || data?.error || `Upload failed (${res.status})`;
      throw new Error(msg);
    }

    // Replace the "Uploading..." message with success text
    replaceMessage(
      uploadStatusId,
      `✅ PDF uploaded${data?.name ? `: ${data.name}` : ""}. You can start chatting about it.`
    );
  } catch (e) {
    // Replace status line with failure info
    replaceMessage(uploadStatusId, `❌ PDF upload failed: ${e.message}`);
  } finally {
    // Re-enable UI, persist messages, scroll
    setBusy({ uploading: false });
    saveMessages();
    scrollToBottom();
  }
});


/**
 * Clear chat button:
 * - Clears messages in memory + localStorage
 * - Clears chat UI
 * - Shows a small "Chat history cleared" bubble (not persisted)
 */
clearBtn.addEventListener("click", async () => {
  messages = [];
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}

  chatEl.innerHTML = "";

  // Ephemeral UI bubble (NOT saved)
  const row = document.createElement("div");
  row.className = "d-flex mb-2 justify-content-start";

  const bubble = document.createElement("div");
  bubble.className = "bubble system";
  bubble.textContent = "Chat history cleared.";

  row.appendChild(bubble);
  chatEl.appendChild(row);

  scrollToBottom();
});


/**
 * Clean PDF button:
 * - Clears backend PDF/index (/clear)
 * - Resets filename display
 * - Clears chat + localStorage
 * - Re-adds the welcome message
 */
cleanPdfBtn?.addEventListener("click", async () => {
  // 1) Clear backend state
  try {
    await fetch(ENDPOINTS.clear, { method: "POST" });
  } catch (e) {
    console.warn("Failed to clear backend state:", e);
  }

  // 2) Reset selected PDF pill
  filePill.textContent = "No PDF selected";
  filePill.title = "";

  // 3) Clear chat history
  messages = [];
  try { localStorage.removeItem(STORAGE_KEY); } catch (_) {}
  chatEl.innerHTML = "";

  // 4) Show welcome message
  addMessage("system", "👋 Welcome! Please upload a PDF to begin chatting.");
  scrollToBottom();
});


/* ---------------------------------------------------------
 * 5) Chat sending logic
 * --------------------------------------------------------- */

// Send button click -> send message
sendBtn.addEventListener("click", () => sendUserMessage());

// Press Enter to send (Shift+Enter makes newline)
textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendUserMessage();
  }
});

/**
 * Collect the user input text and push it into message list,
 * then call backend for assistant reply.
 */
function sendUserMessage() {
  const text = (textInput.value || "").trim();
  if (!text) return;

  // Clear input immediately
  textInput.value = "";

  // Add user message to state/UI
  addMessage("user", text);
  scrollToBottom();

  // Call backend to get assistant response
  askBackendForReply().catch(() => {});
}

/**
 * Calls /chat endpoint:
 * - Adds a typing placeholder "…"
 * - Sends messages to backend
 * - Replaces placeholder with assistant reply
 */
async function askBackendForReply() {
  setSending(true);

  // Placeholder assistant bubble while waiting for backend response
  const typingId = addMessage("assistant", "…");
  scrollToBottom();

  try {
    // IMPORTANT: exclude the typing placeholder from payload
    const payloadMessages = messages.filter(m => m.id !== typingId);

    const res = await fetch(ENDPOINTS.chat, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: payloadMessages })
    });

    if (!res.ok) throw new Error("Chat failed");

    const data = await res.json();

    // Backend is expected to return { reply: "..." }
    const reply = (data && data.reply) ? String(data.reply) : "(No reply returned)";

    // Replace "…" bubble with the actual response
    replaceMessage(typingId, reply);
  } catch (e) {
    // If backend is unreachable or returns error
    replaceMessage(
      typingId,
      "⚠️ Could not reach /chat. Implement the backend endpoint or update ENDPOINTS.chat."
    );
  } finally {
    // Persist + restore UI
    saveMessages();
    setSending(false);
    scrollToBottom();
  }
}

/**
 * Simple helper: disable buttons while sending chat.
 * (This is separate from setBusy; you can merge them if you want.)
 */
function setSending(isSending) {
  sendBtn.disabled = isSending;
  uploadBtn.disabled = isSending;
  clearBtn.disabled = isSending;

  // ✅ also disable “Clean the uploaded PDF”
  if (cleanPdfBtn) cleanPdfBtn.disabled = isSending;
}


/* ---------------------------------------------------------
 * 6) Rendering messages
 * --------------------------------------------------------- */

/**
 * Re-renders all messages from scratch.
 * Called when we update/replace a message.
 */
function renderAll() {
  chatEl.innerHTML = "";
  messages.forEach(renderMessage);
}

/**
 * Render a single message bubble.
 * Applies different alignment and styles for user/system/assistant.
 */
function renderMessage(m) {
  const row = document.createElement("div");
  row.className =
    "d-flex mb-2 " +
    (m.role === "user" ? "justify-content-end" : "justify-content-start");

  const bubble = document.createElement("div");
  bubble.className = "bubble " + m.role;

  // While uploading, attach a small spinner next to upload status message
  if (
    m.role === "system" &&
    busy.uploading &&
    String(m.content).startsWith("Uploading PDF:")
  ) {
    bubble.textContent = m.content + " ";

    const sp = document.createElement("span");
    sp.className = "spinner-border spinner-border-sm align-middle";
    sp.setAttribute("role", "status");
    sp.setAttribute("aria-hidden", "true");

    bubble.appendChild(sp);

  // "Typing" placeholder: show animated dots instead of literal "…"
  } else if (m.role === "assistant" && String(m.content).trim() === "…") {
    const typing = document.createElement("div");
    typing.className = "typing";

    for (let i = 0; i < 3; i++) {
      const dot = document.createElement("span");
      dot.className = "dot";
      typing.appendChild(dot);
    }

    bubble.appendChild(typing);
  } else {
    // Normal message text
    bubble.textContent = m.content;
  }

  row.appendChild(bubble);
  chatEl.appendChild(row);
}


/* ---------------------------------------------------------
 * 7) Message helpers
 * --------------------------------------------------------- */

/**
 * Adds a message to our state and renders it immediately.
 * Returns the message id so we can update it later.
 */
function addMessage(role, content) {
  const id = crypto.randomUUID();
  const msg = { id, role, content: String(content ?? "") };

  messages.push(msg);
  renderMessage(msg);

  saveMessages();
  return id;
}

/**
 * Replace message content by id (used for status updates and replies)
 * Then re-render the whole chat for correctness.
 */
function replaceMessage(id, newContent) {
  const idx = messages.findIndex(m => m.id === id);
  if (idx === -1) return;

  messages[idx].content = String(newContent ?? "");
  renderAll();
}

/**
 * Unified UI busy state handler:
 * - disable UI controls when uploading/thinking
 * - show/hide spinners
 */
function setBusy(next) {
  if (next && typeof next === "object") {
    if (typeof next.uploading === "boolean") busy.uploading = next.uploading;
    if (typeof next.thinking === "boolean") busy.thinking = next.thinking;
  }

  const isBusy = busy.uploading || busy.thinking;

  // Disable UI controls when busy
  sendBtn.disabled = isBusy;
  uploadBtn.disabled = isBusy;
  clearBtn.disabled = isBusy;
  textInput.disabled = isBusy;

  // ✅ also disable “Clean the uploaded PDF”
  if (cleanPdfBtn) cleanPdfBtn.disabled = isBusy;

  // Toggle spinners
  uploadSpinner.classList.toggle("d-none", !busy.uploading);
  sendSpinner.classList.toggle("d-none", !busy.thinking);
}


/* ---------------------------------------------------------
 * 8) Local storage
 * --------------------------------------------------------- */

/**
 * Save messages array to localStorage (best-effort).
 */
function saveMessages() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  } catch (_) {}
}

/**
 * Load messages from localStorage (best-effort).
 * Ensures each message has {id, role, content}.
 */
function loadMessages() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];

    const arr = JSON.parse(raw);
    if (!Array.isArray(arr)) return [];

    return arr.map(m => ({
      id: m.id || crypto.randomUUID(),
      role: m.role || "assistant",
      content: String(m.content || "")
    }));
  } catch (_) {
    return [];
  }
}


/* ---------------------------------------------------------
 * 9) Scrolling
 * --------------------------------------------------------- */

/**
 * Scroll the chat container to the bottom.
 * Call this after adding new messages.
 */
function scrollToBottom() {
  chatEl.scrollTop = chatEl.scrollHeight;
}
