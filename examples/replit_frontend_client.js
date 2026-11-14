// Minimal browser-side client example for a future Replit React/Vite UI.
// Set VITE_API_BASE_URL to the deployed FastAPI backend URL.
const API = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function createSession(language = "en") {
  const r = await fetch(`${API}/api/session?language=${encodeURIComponent(language)}`, { method: "POST" });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function sendChat({ session_id, message, language, profile }) {
  const r = await fetch(`${API}/api/chat`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id, message, language, profile })
  });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function sendVoiceTurn({ file, language = "auto", session_id = "", profile = {} }) {
  const body = new FormData();
  body.append("file", file);
  body.append("language", language);
  body.append("session_id", session_id);
  body.append("profile_json", JSON.stringify(profile));
  const r = await fetch(`${API}/api/voice/turn`, { method: "POST", body });
  if (!r.ok) throw new Error(await r.text());
  const data = await r.json();
  const audio = new Audio(`data:${data.audio_mime_type};base64,${data.reply_audio_base64}`);
  await audio.play();
  return data;
}
