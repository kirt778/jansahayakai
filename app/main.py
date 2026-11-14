import os, uuid, logging, json, base64
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, HTMLResponse, FileResponse
from app.contracts import ChatRequest, EligibilityRequest, SpeechRequest, ProfilePatch
from app.core.engine import run_eligibility, list_schemes, get_scheme, get_csc_info, search_scheme_rag
from app.core.dialogue import make_response
from app.core.voice import transcribe_audio, translate_text, synthesize_speech, resolve_sample_file, LANG_MAP

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("scheme-agent")

app = FastAPI(
    title="JanSahayak — Multilingual Government Scheme Voice AI Agent",
    version="3.0.0",
    description="Voice-first multilingual assistant grounded in official myScheme data."
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=False,
    allow_methods=["*"], allow_headers=["*"]
)

SESSIONS = {}
LANG_CODES = {"en": "en", "hi": "hi", "ta": "ta", "kn": "kn", "pa": "pa", "bn": "bn"}

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "scheme-voice-agent", "version": app.version, "schemes_loaded": len(list_schemes())}

@app.get("/api/languages")
def languages():
    return {"languages": [
        {"code": "hi", "name": "Hindi (हिन्दी)", "sample_file": "Hindi.m4a"},
        {"code": "ta", "name": "Tamil (தமிழ்)", "sample_file": "Tamil.m4a"},
        {"code": "kn", "name": "Kannada (ಕನ್ನಡ)", "sample_file": "Kannada.m4a"},
        {"code": "pa", "name": "Punjabi (ਪੰਜਾਬੀ)", "sample_file": "Punjabi.m4a"},
        {"code": "bn", "name": "Bengali (বাংলা)", "sample_file": None},
        {"code": "en", "name": "English (India)", "sample_file": None}
    ]}

@app.get("/api/samples/{filename}")
def get_person2_sample_audio(filename: str):
    target = resolve_sample_file(filename)
    if target and target.exists():
        media = "audio/mp4" if target.name.lower().endswith(".m4a") else "audio/mpeg"
        return FileResponse(str(target), media_type=media)
    raise HTTPException(status_code=404, detail="Sample audio not found.")

@app.get("/api/schemes")
def schemes():
    all_s = list_schemes()
    return {"count": len(all_s), "schemes": [
        {
            "scheme_id": s["scheme_id"],
            "name": s["scheme_name"],
            "ministry": s.get("ministry", ""),
            "state_specific": s.get("state_specific", "All India"),
            "benefits_summary": s.get("benefits_summary", ""),
            "source_url": s["source_url"],
            "date_checked": s["date_checked"]
        }
        for s in all_s
    ]}

@app.post("/api/session")
def create_session(language: str = "en"):
    lang = LANG_MAP.get(language.lower(), "en")
    sid = str(uuid.uuid4())
    SESSIONS[sid] = {"profile": {}, "language": lang, "last_asked_field": "state"}
    greetings = {
        "en": "Hello! I am JanSahayak. I will ask you 7 short step-by-step questions to find your eligible government schemes. Step 1 of 7: Which state do you live in?",
        "hi": "नमस्ते! मैं जनसहायक हूँ। आपकी पात्र सरकारी योजनाएं खोजने के लिए मैं आपसे एक-एक करके 7 छोटे प्रश्न पूछूँगा। [चरण 1/7] आप किस राज्य में रहते हैं?",
        "ta": "வணக்கம்! நான் ஜன்சஹாயக். உங்களுக்கான அரசுத் திட்டங்களைக் கண்டறிய 7 சிறிய கேள்விகளை படிப்படியாகக் கேட்பேன். [படி 1/7] நீங்கள் எந்த மாநிலத்தில் வசிக்கிறீர்கள்?",
        "kn": "ನಮಸ್ಕಾರ! ನಾನು ಜನಸಹಾಯಕ್. ನಿಮಗೆ ಅರ್ಹವಾದ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳನ್ನು ಹುಡುಕಲು ನಾನು ಹಂತ-ಹಂತವಾಗಿ 7 ಸಣ್ಣ ಪ್ರಶ್ನೆಗಳನ್ನು ಕೇಳುತ್ತೇನೆ. [ಹಂತ 1/7] ನೀವು ಯಾವ ರಾಜ್ಯದಲ್ಲಿ ವಾಸಿಸುತ್ತಿದ್ದೀರಿ?",
        "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਜਨਸਹਾਇਕ ਹਾਂ। ਤੁਹਾਡੀਆਂ ਯੋਗ ਸਰਕਾਰੀ ਸਕੀਮਾਂ ਲੱਭਣ ਲਈ ਮੈਂ ਕਦਮ-ਦਰ-ਕਦਮ 7 ਛੋਟੇ ਸਵਾਲ ਪੁੱਛਾਂਗਾ। [ਪੜਾਅ 1/7] ਤੁਸੀਂ ਕਿਸ ਰਾਜ ਵਿੱਚ ਰਹਿੰਦੇ ਹੋ?",
        "bn": "নমস্কার! আমি জনসহায়ক। আপনার যোগ্য সরকারি প্রকল্পগুলি খুঁজে পেতে আমি ধাপে ধাপে ৭টি ছোট প্রশ্ন করব। [ধাপ 1/7] আপনি কোন রাজ্যে বাস করেন?"
    }
    return {
        "session_id": sid,
        "profile": {},
        "language": lang,
        "stage": "collecting_profile",
        "reply": greetings.get(lang, greetings["en"])
    }

@app.post("/api/chat")
def chat(payload: ChatRequest):
    sid = payload.session_id
    lang = LANG_MAP.get((payload.language or "en").lower(), "en")
    if not sid or sid not in SESSIONS:
        sid = sid or str(uuid.uuid4())
        SESSIONS[sid] = {"profile": {}, "language": lang, "last_asked_field": "state"}
    session = SESSIONS[sid]
    session["language"] = lang

    if payload.profile:
        patch = payload.profile.model_dump(exclude_unset=True)
        session["profile"].update({k: v for k, v in patch.items() if v is not None})

    english_msg = translate_text(payload.message, lang, "en") if lang != "en" else payload.message
    combined_msg = f"{payload.message} {english_msg}"

    result = make_response(
        session["profile"],
        message=combined_msg,
        language=lang,
        last_asked_field=session.get("last_asked_field")
    )
    session["last_asked_field"] = result.get("next_field")

    # Always synthesize dynamic TTS so ALL eligible scheme names are read out loud!
    audio_b64 = ""
    try:
        audio_bytes = synthesize_speech(result["reply"], lang, prefer_bundled_reply=False)
        if audio_bytes:
            audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    except Exception:
        pass

    return {
        "session_id": sid,
        "language": lang,
        "profile": session["profile"],
        "reply_audio_base64": audio_b64,
        "audio_mime_type": "audio/mpeg",
        **result
    }

@app.post("/api/voice/turn")
async def voice_turn(
    file: UploadFile = File(...),
    language: str = Form("auto"),
    session_id: str = Form(""),
    profile_json: str = Form("{}")
):
    audio = await file.read()
    if not audio:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    patch_data = json.loads(profile_json or "{}")
    patch = ProfilePatch(**patch_data)

    try:
        transcript = transcribe_audio(audio, file.filename or "recording.wav", None if language == "auto" else language)
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=str(e))

    detected = LANG_MAP.get(str(transcript.get("language") or language).lower(), "en")
    if language != "auto" and language in LANG_CODES:
        detected = language

    raw_text = transcript.get("text", "")
    english_text = transcript.get("english_text") or (
        translate_text(raw_text, detected, "en") if detected != "en" else raw_text
    )

    sid = session_id or str(uuid.uuid4())
    if sid not in SESSIONS:
        SESSIONS[sid] = {"profile": {}, "language": detected, "last_asked_field": "state"}
    session = SESSIONS[sid]
    session["language"] = detected
    session["profile"].update({k: v for k, v in patch.model_dump(exclude_unset=True).items() if v is not None})

    dialogue = make_response(
        session["profile"],
        message=f"{raw_text} {english_text}",
        language=detected,
        last_asked_field=session.get("last_asked_field")
    )
    session["last_asked_field"] = dialogue.get("next_field")

    # Synthesize fresh speech containing ALL eligible scheme names
    audio_reply = synthesize_speech(dialogue["reply"], detected, prefer_bundled_reply=False)
    return {
        "session_id": sid,
        "detected_language": detected,
        "transcript": raw_text,
        "normalized_text": english_text,
        "reply": dialogue["reply"],
        "english_reply": dialogue.get("english_reply", dialogue["reply"]),
        "reply_audio_base64": base64.b64encode(audio_reply).decode("ascii") if audio_reply else "",
        "audio_mime_type": "audio/mpeg",
        "profile": session["profile"],
        "extracted_this_turn": dialogue.get("extracted_this_turn", {}),
        "stage": dialogue["stage"],
        "profile_complete": dialogue["profile_complete"],
        "matches": dialogue["matches"],
        "rejected_matches": dialogue.get("rejected_matches", []),
        "rag_chunks": dialogue.get("rag_chunks", []),
        "csc_guidance": dialogue.get("csc_guidance"),
        "disclaimer": dialogue["disclaimer"]
    }

@app.post("/api/voice/sample_turn/{sample_name}")
def run_bundled_sample_turn(sample_name: str, session_id: str = ""):
    safe_name = Path(sample_name).name
    sample_path = resolve_sample_file(safe_name)
    audio = sample_path.read_bytes() if (sample_path and sample_path.exists()) else b"sample"
    transcript = transcribe_audio(audio, safe_name, "auto")
    detected = transcript["language"]
    raw_text = transcript["text"]
    english_text = transcript.get("english_text") or translate_text(raw_text, detected, "en")

    sid = session_id or str(uuid.uuid4())
    SESSIONS[sid] = {"profile": {}, "language": detected, "last_asked_field": None}
    session = SESSIONS[sid]

    dialogue = make_response(session["profile"], message=f"{raw_text} {english_text}", language=detected)
    # Synthesize dynamic TTS so it reads out the names of ALL eligible schemes!
    audio_reply = synthesize_speech(dialogue["reply"], detected, prefer_bundled_reply=False)

    return {
        "session_id": sid,
        "sample_file": safe_name,
        "detected_language": detected,
        "transcript": raw_text,
        "normalized_text": english_text,
        "reply": dialogue["reply"],
        "english_reply": dialogue.get("english_reply", ""),
        "reply_audio_base64": base64.b64encode(audio_reply).decode("ascii") if audio_reply else "",
        "audio_mime_type": "audio/mpeg",
        "profile": session["profile"],
        "extracted_this_turn": dialogue.get("extracted_this_turn", {}),
        "stage": dialogue["stage"],
        "profile_complete": dialogue["profile_complete"],
        "matches": dialogue["matches"],
        "rejected_matches": dialogue.get("rejected_matches", []),
        "rag_chunks": dialogue.get("rag_chunks", []),
        "csc_guidance": dialogue.get("csc_guidance"),
        "disclaimer": dialogue["disclaimer"]
    }

@app.post("/api/voice/speak")
def voice_speak(payload: SpeechRequest):
    code = LANG_MAP.get(payload.language.lower(), "en")
    audio = synthesize_speech(payload.text, code)
    return Response(content=audio, media_type="audio/mpeg")

@app.delete("/api/session/{session_id}")
def delete_session(session_id: str):
    SESSIONS.pop(session_id, None)
    return {"deleted": True}

FRONTEND_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>JanSahayak AI — Multilingual Voice Agent for Government Schemes</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; }
    .glass { background: rgba(30, 41, 59, 0.78); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.09); }
    .pulse-ring { animation: pulseRing 1.5s infinite; }
    @keyframes pulseRing {
      0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
      70% { box-shadow: 0 0 0 18px rgba(239, 68, 68, 0); }
      100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <div class="bg-gradient-to-r from-amber-500/20 via-emerald-500/20 to-sky-500/20 border-b border-white/10 px-4 py-2 text-xs text-center text-slate-200 flex items-center justify-center gap-2 flex-wrap">
    <span id="ui_unofficial_badge" class="px-2 py-0.5 rounded-full bg-amber-500/30 border border-amber-400/40 text-amber-200 font-semibold">Unofficial Helper</span>
    <span id="ui_top_banner">Grounded in official <b>myScheme.gov.in</b> data (35 Central & State Schemes). Zero hallucinated rules. Never asks for Aadhaar, Bank A/C, or OTPs.</span>
  </div>

  <header class="glass sticky top-0 z-30 px-6 py-3.5 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800">
    <div class="flex items-center gap-3">
      <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-orange-500 via-white to-emerald-600 flex items-center justify-center shadow-lg font-extrabold text-slate-900 text-lg">ज</div>
      <div>
        <h1 class="text-lg font-bold tracking-tight text-white flex items-center gap-2">
          <span id="ui_app_title">JanSahayak Voice AI</span>
          <span class="text-xs px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-medium">PS-05 GovTech</span>
        </h1>
        <p id="ui_app_subtitle" class="text-xs text-slate-400">Multilingual Voice Assistant + Deterministic Rule Engine + Neural RAG</p>
      </div>
    </div>

    <div class="flex items-center gap-3 flex-wrap">
      <div class="flex items-center gap-2 bg-slate-800/90 border border-slate-700 rounded-xl px-3 py-1.5">
        <span id="ui_lang_label" class="text-xs text-slate-400">Language:</span>
        <select id="langSelect" onchange="onLanguageChange()" class="bg-transparent text-sm font-semibold text-white focus:outline-none">
          <option value="hi">🇮🇳 Hindi (हिन्दी)</option>
          <option value="ta">🇮🇳 Tamil (தமிழ்)</option>
          <option value="kn">🇮🇳 Kannada (ಕನ್ನಡ)</option>
          <option value="pa">🇮🇳 Punjabi (ਪੰਜਾਬੀ)</option>
          <option value="bn">🇮🇳 Bengali (বাংলা)</option>
          <option value="en">🇮🇳 English (India)</option>
        </select>
      </div>

      <label class="flex items-center gap-2 text-xs bg-slate-800/80 border border-slate-700 px-3 py-2 rounded-xl cursor-pointer hover:border-slate-500">
        <input type="checkbox" id="lowBwToggle" class="accent-emerald-500" />
        <span id="ui_low_bw">Low-Bandwidth Mode</span>
      </label>

      <button id="ui_reset_btn" onclick="resetConversation()" class="text-xs bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/30 px-3 py-2 rounded-xl font-medium transition">
        Reset Session
      </button>
    </div>
  </header>

  <section class="bg-slate-900/90 border-b border-slate-800 px-6 py-3">
    <div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
      <div class="flex items-center gap-2 flex-wrap">
        <span id="ui_demo_bar_label" class="text-xs font-semibold uppercase tracking-wider text-amber-400 mr-1">🎙️ Run Bundled Voice Files (person2/*.m4a):</span>
        <button onclick="runBundledSample('Hindi.m4a', 'hi')" class="text-xs bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-400/40 text-indigo-200 px-3 py-1.5 rounded-lg font-medium transition">▶ Hindi.m4a (UP Farmer)</button>
        <button onclick="runBundledSample('Tamil.m4a', 'ta')" class="text-xs bg-emerald-600/30 hover:bg-emerald-600/50 border border-emerald-400/40 text-emerald-200 px-3 py-1.5 rounded-lg font-medium transition">▶ Tamil.m4a (TN Farmer)</button>
        <button onclick="runBundledSample('Kannada.m4a', 'kn')" class="text-xs bg-amber-600/30 hover:bg-amber-600/50 border border-amber-400/40 text-amber-200 px-3 py-1.5 rounded-lg font-medium transition">▶ Kannada.m4a (KA Farmer)</button>
        <button onclick="runBundledSample('Punjabi.m4a', 'pa')" class="text-xs bg-purple-600/30 hover:bg-purple-600/50 border border-purple-400/40 text-purple-200 px-3 py-1.5 rounded-lg font-medium transition">▶ Punjabi.m4a (PB Farmer)</button>
      </div>
      <div class="flex items-center gap-2 flex-wrap">
        <button onclick="runPersonaPreset('bengali_woman')" class="text-xs bg-sky-600/30 hover:bg-sky-600/50 border border-sky-400/40 text-sky-200 px-3 py-1.5 rounded-lg font-medium transition">✨ Bengali Persona (WB Woman)</button>
        <button id="ui_oos_btn" onclick="testOutOfScope()" class="text-xs bg-rose-600/30 hover:bg-rose-600/50 border border-rose-400/40 text-rose-200 px-3 py-1.5 rounded-lg font-medium transition">🛡️ Test Out-of-Scope Refusal</button>
      </div>
    </div>
  </section>

  <main class="max-w-7xl w-full mx-auto flex-1 grid grid-cols-1 lg:grid-cols-12 gap-5 p-5">
    <div class="lg:col-span-3 space-y-4">
      <div class="glass rounded-2xl p-4">
        <div class="flex items-center justify-between mb-3">
          <h2 id="ui_profile_title" class="text-sm font-bold text-white flex items-center gap-2">👤 Citizen Profile State</h2>
          <span id="stageBadge" class="text-[11px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">Collecting</span>
        </div>
        <p id="ui_profile_desc" class="text-xs text-slate-400 mb-3">Extracted automatically from your spoken conversation (or edit below):</p>

        <div class="space-y-2.5 text-xs">
          <div>
            <label id="ui_lbl_state" class="text-slate-400 block mb-1">State / UT</label>
            <input id="prof_state" type="text" placeholder="e.g. Karnataka, Tamil Nadu, UP" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white" />
          </div>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label id="ui_lbl_age" class="text-slate-400 block mb-1">Age (Years)</label>
              <input id="prof_age" type="number" placeholder="e.g. 40" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white" />
            </div>
            <div>
              <label id="ui_lbl_gender" class="text-slate-400 block mb-1">Gender</label>
              <select id="prof_gender" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2 py-1.5 text-white">
                <option value="">--</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>
          <div>
            <label id="ui_lbl_occ" class="text-slate-400 block mb-1">Occupation</label>
            <select id="prof_occupation" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2 py-1.5 text-white">
              <option value="">--</option>
              <option value="farmer">Farmer / Agriculture</option>
              <option value="student">Student</option>
              <option value="street_vendor">Street Vendor / Hawker</option>
              <option value="artisan">Artisan / Vishwakarma</option>
              <option value="unemployed">Unemployed / Homemaker</option>
              <option value="entrepreneur">Entrepreneur / Self-Employed</option>
              <option value="laborer">Unorganized Worker / Laborer</option>
            </select>
          </div>
          <div class="grid grid-cols-2 gap-2">
            <div>
              <label id="ui_lbl_income" class="text-slate-400 block mb-1">Annual Income (₹)</label>
              <input id="prof_income" type="number" placeholder="e.g. 85000" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white" />
            </div>
            <div>
              <label id="ui_lbl_cat" class="text-slate-400 block mb-1">Category</label>
              <select id="prof_category" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2 py-1.5 text-white">
                <option value="">--</option>
                <option value="General">General</option>
                <option value="OBC">OBC</option>
                <option value="SC">SC</option>
                <option value="ST">ST</option>
              </select>
            </div>
          </div>
          <div>
            <label id="ui_lbl_land" class="text-slate-400 block mb-1">Owns Cultivable Land?</label>
            <select id="prof_land" class="w-full bg-slate-900/90 border border-slate-700 rounded-lg px-2 py-1.5 text-white">
              <option value="">Unknown</option>
              <option value="true">Yes (Owns Land)</option>
              <option value="false">No (Landless)</option>
            </select>
          </div>
          <button id="ui_update_prof_btn" onclick="applyManualProfile()" class="w-full mt-2 bg-emerald-600 hover:bg-emerald-500 text-white font-semibold py-2 rounded-xl transition shadow">
            ✓ Update Profile & Evaluate Rules
          </button>
        </div>
      </div>

      <div class="glass rounded-2xl p-4 border border-sky-500/20">
        <h3 id="ui_csc_title" class="text-xs font-bold uppercase tracking-wider text-sky-400 mb-1.5">🏢 Nearest Common Service Centre (CSC)</h3>
        <p id="cscName" class="text-sm font-semibold text-white">Digital Seva Common Service Centre (CSC)</p>
        <p id="cscInstr" class="text-xs text-slate-300 mt-1">Visit with original documents from the checklist for assisted application submission.</p>
        <div class="mt-2.5 flex items-center justify-between text-xs">
          <span class="text-emerald-400 font-medium">📞 Helpline: 14599</span>
          <a id="ui_csc_link" href="https://locator.csccloud.in/" target="_blank" class="text-sky-400 hover:underline font-semibold">Locate CSC ↗</a>
        </div>
      </div>
    </div>

    <div class="lg:col-span-5 flex flex-col glass rounded-2xl overflow-hidden h-[680px]">
      <div class="bg-emerald-950/60 border-b border-emerald-500/20 px-4 py-3 flex items-center justify-between">
        <div class="flex items-center gap-2.5">
          <div class="w-3 h-3 rounded-full bg-emerald-400 animate-pulse"></div>
          <div>
            <h2 id="ui_chat_title" class="text-sm font-bold text-white">Voice & Chat Assistant</h2>
            <p id="ui_chat_subtitle" class="text-[11px] text-emerald-300">Speaks & listens in Hindi, Tamil, Kannada, Punjabi, Bengali & English</p>
          </div>
        </div>
        <audio id="audioPlayer" controls class="h-7 w-40"></audio>
      </div>

      <div id="chatBox" class="flex-1 overflow-y-auto p-4 space-y-3 bg-slate-950/40"></div>

      <div class="p-3 bg-slate-900/95 border-t border-slate-800 space-y-2">
        <div class="flex items-center gap-2">
          <button id="micBtn" onclick="toggleMicRecording()" class="flex items-center gap-2 bg-rose-600 hover:bg-rose-500 text-white text-xs font-bold px-3.5 py-2.5 rounded-xl transition shadow-lg">
            <span id="micIcon">🎙️</span>
            <span id="micLabel">Speak Now</span>
          </button>

          <label class="flex items-center gap-1.5 bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs font-medium px-3 py-2.5 rounded-xl cursor-pointer transition">
            <span id="ui_upload_btn">📁 Upload Audio (.m4a/.mp3)</span>
            <input type="file" id="audioUploadInput" accept=".m4a,.mp3,.wav,.webm,.ogg,audio/*" class="hidden" onchange="uploadAudioFile(this)" />
          </label>

          <span id="voiceStatus" class="text-[11px] text-slate-400 truncate">Ready for voice or text</span>
        </div>

        <div class="flex items-center gap-2">
          <input id="chatInput" type="text" placeholder="Type in any language or code-mixed Hindi-English..."
            class="flex-1 bg-slate-800/90 border border-slate-700 rounded-xl px-3.5 py-2 text-sm text-white focus:outline-none focus:border-emerald-500"
            onkeydown="if(event.key==='Enter') sendTextTurn()" />
          <button id="ui_send_btn" onclick="sendTextTurn()" class="bg-emerald-600 hover:bg-emerald-500 text-white font-semibold px-4 py-2 rounded-xl text-sm transition">
            Send
          </button>
        </div>
      </div>
    </div>

    <div class="lg:col-span-4 flex flex-col glass rounded-2xl overflow-hidden h-[680px]">
      <div class="p-4 border-b border-slate-800 flex items-center justify-between">
        <div>
          <h2 id="ui_schemes_title" class="text-sm font-bold text-white">📋 Deterministic Scheme Matches</h2>
          <p id="ui_schemes_subtitle" class="text-[11px] text-slate-400">Rule Engine Decisions + Grounded Official Citations</p>
        </div>
        <div class="flex gap-1 bg-slate-900 p-1 rounded-xl border border-slate-800 text-xs">
          <button id="tabEligibleBtn" onclick="switchSchemeTab('eligible')" class="px-2.5 py-1 rounded-lg bg-emerald-600 text-white font-semibold"><span id="ui_tab_elig">Eligible</span> (<span id="eligCount">0</span>)</button>
          <button id="tabRejectedBtn" onclick="switchSchemeTab('rejected')" class="px-2.5 py-1 rounded-lg text-slate-400 hover:text-white"><span id="ui_tab_rej">Why Not Eligible?</span> (<span id="rejCount">0</span>)</button>
        </div>
      </div>

      <div id="schemesContainer" class="flex-1 overflow-y-auto p-4 space-y-3"></div>
    </div>
  </main>

  <script>
    let sessionId = "";
    let currentMatches = [];
    let currentRejected = [];
    let activeTab = "eligible";
    let isRecording = false;
    let recTimerInterval = null;
    let recSeconds = 0;
    let browserSpeechText = "";
    let activeSpeechRec = null;
    let audioContext = null;
    let mediaStream = null;
    let scriptProcessor = null;
    let pcmChunks = [];
    let pendingReplyTimeout = null;

    const SPEECH_LANG_MAP = {
      hi: "hi-IN", ta: "ta-IN", kn: "kn-IN", pa: "pa-IN", bn: "bn-IN", en: "en-IN"
    };

    // Complete UI Screen Localization Dictionary for all 6 languages
    const UI_STRINGS = {
      en: {
        unofficial_badge: "Unofficial Helper",
        top_banner: "Grounded in official <b>myScheme.gov.in</b> data (35 Central & State Schemes). Zero hallucinated rules. Never asks for Aadhaar, Bank A/C, or OTPs.",
        app_title: "JanSahayak Voice AI",
        app_subtitle: "Multilingual Voice Assistant + Deterministic Rule Engine + Neural RAG",
        lang_label: "Language:",
        low_bw: "Low-Bandwidth Mode",
        reset_btn: "Reset Session",
        demo_bar_label: "🎙️ Run Bundled Voice Files (person2/*.m4a):",
        oos_btn: "🛡️ Test Out-of-Scope Refusal",
        profile_title: "👤 Citizen Profile State",
        profile_desc: "Extracted automatically from your spoken conversation (or edit below):",
        lbl_state: "State / UT",
        lbl_age: "Age (Years)",
        lbl_gender: "Gender",
        lbl_occ: "Occupation",
        lbl_income: "Annual Income (₹)",
        lbl_cat: "Category",
        lbl_land: "Owns Cultivable Land?",
        update_prof_btn: "✓ Update Profile & Evaluate Rules",
        csc_title: "🏢 Nearest Common Service Centre (CSC)",
        csc_link: "Locate CSC ↗",
        chat_title: "Voice & Chat Assistant",
        chat_subtitle: "Speaks & listens in Hindi, Tamil, Kannada, Punjabi, Bengali & English",
        mic_speak: "Speak Now",
        upload_btn: "📁 Upload Audio (.m4a/.mp3)",
        chat_placeholder: "Type in any language or code-mixed Hindi-English...",
        send_btn: "Send",
        schemes_title: "📋 Deterministic Scheme Matches",
        schemes_subtitle: "Rule Engine Decisions + Grounded Official Citations",
        tab_elig: "Eligible",
        tab_rej: "Why Not Eligible?",
        badge_elig: "✓ Eligible",
        badge_poss: "⏳ Possibly Eligible",
        badge_rej: "✗ Not Eligible",
        rule_verif: "✅ Rule Engine Verification:",
        why_not: "❌ Why Not Eligible:",
        checklist_summary: "📄 Document Checklist & Application Steps",
        req_docs: "Required Documents Checklist:",
        how_to_apply: "How to Apply Step-by-Step:",
        step_word: "Step",
        verified_word: "Verified:",
        source_link: "🔗 Official Source Citation ↗",
        empty_schemes: "Speak into the microphone, click a bundled .m4a file above, or enter your details to see matched schemes in your language."
      },
      hi: {
        unofficial_badge: "गैर-सरकारी सहायक",
        top_banner: "आधिकारिक <b>myScheme.gov.in</b> डेटा (35 केंद्र और राज्य योजनाओं) पर आधारित। कभी भी आधार, बैंक खाता या OTP नहीं मांगता।",
        app_title: "जनसहायक वॉयस एआई (JanSahayak)",
        app_subtitle: "बहुभाषी वॉयस सहायक + नियम इंजन + ग्राउंडेड RAG",
        lang_label: "भाषा:",
        low_bw: "लो-बैंडविड्थ मोड",
        reset_btn: "सेशन रीसेट करें",
        demo_bar_label: "🎙️ सैंपल वॉयस फ़ाइलें चलाएं (person2/*.m4a):",
        oos_btn: "🛡️ दायरे से बाहर प्रश्न परीक्षण",
        profile_title: "👤 नागरिक प्रोफ़ाइल विवरण",
        profile_desc: "आपकी बातचीत से स्वतः निकाला गया विवरण (या नीचे संपादित करें):",
        lbl_state: "राज्य / केंद्र शासित प्रदेश",
        lbl_age: "आयु (वर्ष)",
        lbl_gender: "लिंग",
        lbl_occ: "व्यवसाय (पेशा)",
        lbl_income: "वार्षिक पारिवारिक आय (₹)",
        lbl_cat: "वर्ग (Category)",
        lbl_land: "क्या कृषि भूमि है?",
        update_prof_btn: "✓ प्रोफ़ाइल अपडेट करें और योजनाएं जांचें",
        csc_title: "🏢 निकटतम जन सेवा केंद्र (CSC)",
        csc_link: "CSC केंद्र खोजें ↗",
        chat_title: "वॉयस और चैट सहायक",
        chat_subtitle: "हिन्दी, तमिल, कन्नड़, पंजाबी, बंगाली और अंग्रेजी में संवाद",
        mic_speak: "अब बोलें",
        upload_btn: "📁 ऑडियो अपलोड करें",
        chat_placeholder: "अपनी भाषा या हिंग्लिश में लिखें...",
        send_btn: "भेजें",
        schemes_title: "📋 पात्र सरकारी योजनाएं",
        schemes_subtitle: "नियम इंजन निर्णय + आधिकारिक स्रोत लिंक",
        tab_elig: "पात्र योजनाएं",
        tab_rej: "अपात्र क्यों?",
        badge_elig: "✓ पात्र (Eligible)",
        badge_poss: "⏳ संभावित पात्र",
        badge_rej: "✗ अपात्र",
        rule_verif: "✅ पात्रता नियम सत्यापन:",
        why_not: "❌ अपात्र होने का कारण:",
        checklist_summary: "📄 आवश्यक दस्तावेज़ और आवेदन प्रक्रिया",
        req_docs: "आवश्यक दस्तावेजों की सूची:",
        how_to_apply: "आवेदन करने के चरण:",
        step_word: "चरण",
        verified_word: "सत्यापित:",
        source_link: "🔗 आधिकारिक स्रोत लिंक ↗",
        empty_schemes: "योजनाओं की जानकारी देखने के लिए माइक पर बोलें या ऊपर दिए गए .m4a बटन पर क्लिक करें।"
      },
      ta: {
        unofficial_badge: "அதிகாரப்பூர்வமற்ற உதவியாளர்",
        top_banner: "அதிகாரப்பூர்வ <b>myScheme.gov.in</b> தரவுத்தளத்தை அடிப்படையாகக் கொண்டது (35 திட்டங்கள்). ஆதார், வங்கி எண் அல்லது OTP கேட்கப்படமாட்டாது.",
        app_title: "ஜன்சஹாயக் குரல் AI (JanSahayak)",
        app_subtitle: "பன்மொழி குரல் உதவியாளர் + விதி இயந்திரம் + RAG",
        lang_label: "மொழி:",
        low_bw: "குறைந்த இணைய வேக முறை",
        reset_btn: "மீட்டமை (Reset)",
        demo_bar_label: "🎙️ மாதிரி குரல் கோப்புகள் (person2/*.m4a):",
        oos_btn: "🛡️ வரம்பிற்கு அப்பாற்பட்ட கேள்வி",
        profile_title: "👤 குடிமக்கள் சுயவிவரம்",
        profile_desc: "உங்கள் உரையாடலில் இருந்து தானாகப் பெறப்பட்ட விவரங்கள்:",
        lbl_state: "மாநிலம்",
        lbl_age: "வயது (ஆண்டுகள்)",
        lbl_gender: "பாலினம்",
        lbl_occ: "தொழில்",
        lbl_income: "ஆண்டு வருமானம் (₹)",
        lbl_cat: "சமூகப் பிரிவு",
        lbl_land: "விவசாய நிலம் உள்ளதா?",
        update_prof_btn: "✓ விவரங்களைப் புதுப்பித்து திட்டங்களைக் காண்க",
        csc_title: "🏢 அருகிலுள்ள இ-சேவை / பொது சேவை மையம் (CSC)",
        csc_link: "மையத்தைக் கண்டறிய ↗",
        chat_title: "குரல் மற்றும் அரட்டை உதவியாளர்",
        chat_subtitle: "தமிழ், இந்தி, கன்னடம், பஞ்சாபி, பெங்காலி & ஆங்கிலத்தில் பேசலாம்",
        mic_speak: "இப்போது பேசுங்கள்",
        upload_btn: "📁 ஆடியோ பதிவேற்று",
        chat_placeholder: "உங்கள் மொழியில் தட்டச்சு செய்யவும்...",
        send_btn: "அனுப்பு",
        schemes_title: "📋 தகுதியான அரசுத் திட்டங்கள்",
        schemes_subtitle: "விதி இயந்திர முடிவுகள் + அதிகாரப்பூர்வ ஆதாரங்கள்",
        tab_elig: "தகுதியானவை",
        tab_rej: "ஏன் தகுதியில்லை?",
        badge_elig: "✓ தகுதியானவர்",
        badge_poss: "⏳ தகுதி பெற வாய்ப்பு",
        badge_rej: "✗ தகுதியில்லை",
        rule_verif: "✅ தகுதி விதி சரிபார்ப்பு:",
        why_not: "❌ தகுதியின்மைக்கான காரணம்:",
        checklist_summary: "📄 தேவையான ஆவணங்கள் மற்றும் விண்ணப்பிக்கும் முறை",
        req_docs: "தேவையான ஆவணங்கள் பட்டியல்:",
        how_to_apply: "விண்ணப்பிக்கும் வழிமுறைகள்:",
        step_word: "படி",
        verified_word: "சரிபார்க்கப்பட்டது:",
        source_link: "🔗 அதிகாரப்பூர்வ இணையதளம் ↗",
        empty_schemes: "உங்களுக்கான அரசுத் திட்டங்களைக் காண மைக்கில் பேசவும் அல்லது மேலே உள்ள .m4a பொத்தானை அழுத்தவும்."
      },
      kn: {
        unofficial_badge: "ಅನಧಿಕೃತ ಸಹಾಯಕ",
        top_banner: "ಅಧಿಕೃತ <b>myScheme.gov.in</b> ಮಾಹಿತಿಯ ಆಧಾರದ ಮೇಲೆ (35 ಕೇಂದ್ರ ಮತ್ತು ರಾಜ್ಯ ಯೋಜನೆಗಳು). ಆಧಾರ್, ಬ್ಯಾಂಕ್ ಖಾತೆ ಅಥವಾ OTP ಕೇಳುವುದಿಲ್ಲ.",
        app_title: "ಜನಸಹಾಯಕ್ ಧ್ವನಿ AI (JanSahayak)",
        app_subtitle: "ಬಹುಭಾಷಾ ಧ್ವನಿ ಸಹಾಯಕ + ನಿಯಮ ಎಂಜಿನ್ + RAG",
        lang_label: "ಭಾಷೆ:",
        low_bw: "ಕಡಿಮೆ ಬ್ಯಾಂಡ್‌ವಿಡ್ತ್ ಮೋಡ್",
        reset_btn: "ಸೆಷನ್ ಮರುಹೊಂದಿಸಿ (Reset)",
        demo_bar_label: "🎙️ ಮಾದರಿ ಧ್ವನಿ ಫೈಲ್‌ಗಳನ್ನು ಚಲಾಯಿಸಿ (person2/*.m4a):",
        oos_btn: "🛡️ ವ್ಯಾಪ್ತಿಯ ಹೊರಗಿನ ಪ್ರಶ್ನೆ ಪರೀಕ್ಷೆ",
        profile_title: "👤 ನಾಗರಿಕರ ವಿವರಗಳು",
        profile_desc: "ನಿಮ್ಮ ಸಂಭಾಷಣೆಯಿಂದ ಸ್ವಯಂಚಾಲಿತವಾಗಿ ಸಂಗ್ರಹಿಸಲಾದ ಮಾಹಿತಿ:",
        lbl_state: "ರಾಜ್ಯ",
        lbl_age: "ವಯಸ್ಸು (ವರ್ಷಗಳು)",
        lbl_gender: "ಲಿಂಗ",
        lbl_occ: "ಉದ್ಯೋಗ",
        lbl_income: "ವಾರ್ಷಿಕ ಆದಾಯ (₹)",
        lbl_cat: "ವರ್ಗ (Category)",
        lbl_land: "ಕೃಷಿ ಭೂಮಿ ಇದೆಯೇ?",
        update_prof_btn: "✓ ವಿವರ ನವೀಕರಿಸಿ ಮತ್ತು ಯೋಜನೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ",
        csc_title: "🏢 ಹತ್ತಿರದ ಸಾಮಾನ್ಯ ಸೇವಾ ಕೇಂದ್ರ (CSC / ಕರ್ನಾಟಕ ಒನ್)",
        csc_link: "ಕೇಂದ್ರವನ್ನು ಹುಡುಕಿ ↗",
        chat_title: "ಧ್ವನಿ ಮತ್ತು ಚಾಟ್ ಸಹಾಯಕ",
        chat_subtitle: "ಕನ್ನಡ, ಹಿಂದಿ, ತಮಿಳು, ಪಂಜಾಬಿ, ಬೆಂಗಾಲಿ ಮತ್ತು ಇಂಗ್ಲಿಷ್‌ನಲ್ಲಿ ಮಾತನಾಡಿ",
        mic_speak: "ಈಗ ಮಾತನಾಡಿ",
        upload_btn: "📁 ಆಡಿಯೋ ಅಪ್‌ಲೋಡ್",
        chat_placeholder: "ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಟೈಪ್ ಮಾಡಿ...",
        send_btn: "ಕಳುಹಿಸಿ",
        schemes_title: "📋 ಅರ್ಹ ಸರ್ಕಾರಿ ಯೋಜನೆಗಳು",
        schemes_subtitle: "ನಿಯಮ ಆಧಾರಿತ ನಿರ್ಧಾರಗಳು + ಅಧಿಕೃತ ಮೂಲ ಲಿಂಕ್",
        tab_elig: "ಅರ್ಹ ಯೋಜನೆಗಳು",
        tab_rej: "ಏಕೆ ಅರ್ಹರಲ್ಲ?",
        badge_elig: "✓ ಅರ್ಹರು",
        badge_poss: "⏳ ಭಾಗಶಃ ಅರ್ಹರು",
        badge_rej: "✗ ಅರ್ಹರಲ್ಲ",
        rule_verif: "✅ ಅರ್ಹತಾ ನಿಯಮ ಪರಿಶೀಲನೆ:",
        why_not: "❌ ಅನರ್ಹತೆಗೆ ಕಾರಣ:",
        checklist_summary: "📄 ಅಗತ್ಯ ದಾಖಲೆಗಳು ಮತ್ತು ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ಹಂತಗಳು",
        req_docs: "ಬೇಕಾಗುವ ದಾಖಲೆಗಳ ಪಟ್ಟಿ:",
        how_to_apply: "ಅರ್ಜಿ ಸಲ್ಲಿಸುವ ಹಂತಗಳು:",
        step_word: "ಹಂತ",
        verified_word: "ಪರಿಶೀಲಿಸಲಾಗಿದೆ:",
        source_link: "🔗 ಅಧಿಕೃತ ಮೂಲ ಲಿಂಕ್ ↗",
        empty_schemes: "ಸರ್ಕಾರಿ ಯೋಜನೆಗಳನ್ನು ನೋಡಲು ಮೈಕ್‌ನಲ್ಲಿ ಮಾತನಾಡಿ ಅಥವಾ ಮೇಲಿನ .m4a ಬಟನ್ ಕ್ಲಿಕ್ ಮಾಡಿ."
      },
      pa: {
        unofficial_badge: "ਗੈਰ-ਸਰਕਾਰੀ ਸਹਾਇਕ",
        top_banner: "ਅਧਿਕਾਰਤ <b>myScheme.gov.in</b> ਡੇਟਾ (35 ਕੇਂਦਰੀ ਅਤੇ ਰਾਜ ਸਕੀਮਾਂ) 'ਤੇ ਅਧਾਰਤ। ਕਦੇ ਵੀ ਆਧਾਰ, ਬੈਂਕ ਖਾਤਾ ਜਾਂ OTP ਨਹੀਂ ਮੰਗਦਾ।",
        app_title: "ਜਨਸਹਾਇਕ ਵੌਇਸ ਏਆਈ (JanSahayak)",
        app_subtitle: "ਬਹੁ-ਭਾਸ਼ਾਈ ਵੌਇਸ ਸਹਾਇਕ + ਨਿਯਮ ਇੰਜਣ + RAG",
        lang_label: "ਭਾਸ਼ਾ:",
        low_bw: "ਘੱਟ-ਬੈਂਡਵਿਡਥ ਮੋਡ",
        reset_btn: "ਸੈਸ਼ਨ ਰੀਸੈਟ ਕਰੋ",
        demo_bar_label: "🎙️ ਸੈਂਪਲ ਵੌਇਸ ਫਾਈਲਾਂ ਚਲਾਓ (person2/*.m4a):",
        oos_btn: "🛡️ ਦਾਇਰੇ ਤੋਂ ਬਾਹਰ ਸਵਾਲ ਟੈਸਟ",
        profile_title: "👤 ਨਾਗਰਿਕ ਪ੍ਰੋਫਾਈਲ ਵੇਰਵੇ",
        profile_desc: "ਤੁਹਾਡੀ ਗੱਲਬਾਤ ਤੋਂ ਆਪਣੇ ਆਪ ਲਿਆ ਗਿਆ ਵੇਰਵਾ:",
        lbl_state: "ਰਾਜ / ਪ੍ਰਦੇਸ਼",
        lbl_age: "ਉਮਰ (ਸਾਲ)",
        lbl_gender: "ਲਿੰਗ",
        lbl_occ: "ਕਿੱਤਾ (Occupation)",
        lbl_income: "ਸਾਲਾਨਾ ਪਰਿਵਾਰਕ ਆਮਦਨ (₹)",
        lbl_cat: "ਸ਼੍ਰੇਣੀ (Category)",
        lbl_land: "ਕੀ ਖੇਤੀਬਾੜੀ ਵਾਲੀ ਜ਼ਮੀਨ ਹੈ?",
        update_prof_btn: "✓ ਪ੍ਰੋਫਾਈਲ ਅਪਡੇਟ ਕਰੋ ਅਤੇ ਸਕੀਮਾਂ ਦੇਖੋ",
        csc_title: "🏢 ਨਜ਼ਦੀਕੀ ਸੇਵਾ ਕੇਂਦਰ (CSC / Sewa Kendra)",
        csc_link: "ਕੇਂਦਰ ਲੱਭੋ ↗",
        chat_title: "ਵੌਇਸ ਅਤੇ ਚੈਟ ਸਹਾਇਕ",
        chat_subtitle: "ਪੰਜਾਬੀ, ਹਿੰਦੀ, ਤਮਿਲ, ਕੰਨੜ, ਬੰਗਾਲੀ ਅਤੇ ਅੰਗਰੇਜ਼ੀ ਵਿੱਚ ਗੱਲ ਕਰੋ",
        mic_speak: "ਹੁਣ ਬੋਲੋ",
        upload_btn: "📁 ਆਡੀਓ ਅਪਲੋਡ ਕਰੋ",
        chat_placeholder: "ਆਪਣੀ ਭਾਸ਼ਾ ਵਿੱਚ ਟਾਈਪ ਕਰੋ...",
        send_btn: "ਭੇਜੋ",
        schemes_title: "📋 ਯੋਗ ਸਰਕਾਰੀ ਸਕੀਮਾਂ",
        schemes_subtitle: "ਨਿਯਮ ਇੰਜਣ ਫੈਸਲੇ + ਅਧਿਕਾਰਤ ਸਰੋਤ ਲਿੰਕ",
        tab_elig: "ਯੋਗ ਸਕੀਮਾਂ",
        tab_rej: "ਯੋਗ ਕਿਉਂ ਨਹੀਂ?",
        badge_elig: "✓ ਯੋਗ (Eligible)",
        badge_poss: "⏳ ਸੰਭਾਵਿਤ ਯੋਗ",
        badge_rej: "✗ ਯੋਗ ਨਹੀਂ",
        rule_verif: "✅ ਯੋਗਤਾ ਨਿਯਮ ਪੁਸ਼ਟੀ:",
        why_not: "❌ ਯੋਗ ਨਾ ਹੋਣ ਦਾ ਕਾਰਨ:",
        checklist_summary: "📄 ਲੋੜੀਂਦੇ ਦਸਤਾਵੇਜ਼ ਅਤੇ ਅਰਜ਼ੀ ਦੇ ਪੜਾਅ",
        req_docs: "ਲੋੜੀਂਦੇ ਦਸਤਾਵੇਜ਼ਾਂ ਦੀ ਸੂਚੀ:",
        how_to_apply: "ਅਰਜ਼ੀ ਦੇਣ ਦੇ ਪੜਾਅ:",
        step_word: "ਪੜਾਅ",
        verified_word: "ਪ੍ਰਮਾਣਿਤ:",
        source_link: "🔗 ਅਧਿਕਾਰਤ ਸਰੋਤ ਲਿੰਕ ↗",
        empty_schemes: "ਸਕੀਮਾਂ ਦੇਖਣ ਲਈ ਮਾਈਕ ਵਿੱਚ ਬੋਲੋ ਜਾਂ ਉੱਪਰ ਦਿੱਤੇ .m4a ਬਟਨ 'ਤੇ ਕਲਿੱਕ ਕਰੋ।"
      },
      bn: {
        unofficial_badge: "বেসরকারি সহায়ক",
        top_banner: "অফিসিয়াল <b>myScheme.gov.in</b> তথ্যের উপর ভিত্তি করে (৩৫টি কেন্দ্র ও রাজ্য প্রকল্প)। কখনোই আধার, ব্যাংক অ্যাকাউন্ট বা OTP চায় না।",
        app_title: "জনসহায়ক ভয়েস এআই (JanSahayak)",
        app_subtitle: "বহুভাষিক ভয়েস সহায়ক + নিয়ম ইঞ্জিন + RAG",
        lang_label: "ভাষা:",
        low_bw: "লো-ব্যান্ডউইথ মোড",
        reset_btn: "সেশন রিসেট করুন",
        demo_bar_label: "🎙️ নমুনা ভয়েস ফাইল চালান (person2/*.m4a):",
        oos_btn: "🛡️ আওতার বাইরের প্রশ্ন পরীক্ষা",
        profile_title: "👤 নাগরিক প্রোফাইল বিবরণ",
        profile_desc: "আপনার কথোপকথন থেকে স্বয়ংক্রিয়ভাবে সংগৃহীত তথ্য:",
        lbl_state: "রাজ্য",
        lbl_age: "বয়স (বছর)",
        lbl_gender: "লিঙ্গ",
        lbl_occ: "পেশা",
        lbl_income: "বার্ষিক পারিবারিক আয় (₹)",
        lbl_cat: "বিভাগ (Category)",
        lbl_land: "চাষযোগ্য জমি আছে কি?",
        update_prof_btn: "✓ প্রোফাইল আপডেট করুন ও প্রকল্প দেখুন",
        csc_title: "🏢 নিকটতম কমন সার্ভিস সেন্টার (CSC / দুয়ারে সরকার)",
        csc_link: "কেন্দ্র খুঁজুন ↗",
        chat_title: "ভয়েস এবং চ্যাট সহায়ক",
        chat_subtitle: "বাংলা, হিন্দি, তামিল, কন্নড়, পাঞ্জাবি এবং ইংরেজিতে কথা বলুন",
        mic_speak: "এখন বলুন",
        upload_btn: "📁 অডিও আপলোড",
        chat_placeholder: "আপনার ভাষায় টাইপ করুন...",
        send_btn: "পাঠান",
        schemes_title: "📋 যোগ্য সরকারি প্রকল্পসমূহ",
        schemes_subtitle: "নিয়ম ইঞ্জিন সিদ্ধান্ত + অফিসিয়াল সূত্র লিঙ্ক",
        tab_elig: "যোগ্য প্রকল্প",
        tab_rej: "কেন যোগ্য নয়?",
        badge_elig: "✓ যোগ্য",
        badge_poss: "⏳ সম্ভাব্য যোগ্য",
        badge_rej: "✗ যোগ্য নয়",
        rule_verif: "✅ যোগ্যতা নিয়ম যাচাইকরণ:",
        why_not: "❌ অযোগ্যতার কারণ:",
        checklist_summary: "📄 প্রয়োজনীয় নথিপত্র এবং আবেদনের ধাপ",
        req_docs: "প্রয়োজনীয় নথিপত্রের তালিকা:",
        how_to_apply: "আবেদন করার ধাপসমূহ:",
        step_word: "ধাপ",
        verified_word: "যাচাইকৃত:",
        source_link: "🔗 অফিসিয়াল সূত্র লিঙ্ক ↗",
        empty_schemes: "প্রকল্পগুলি দেখতে মাইকে কথা বলুন অথবা উপরের বাটনে ক্লিক করুন।"
      }
    };


    const DROPDOWN_OPTIONS = {
      en: {
        gender: [["", "--"], ["male", "Male"], ["female", "Female"], ["other", "Other"]],
        occ: [["", "--"], ["farmer", "Farmer / Agriculture"], ["student", "Student"], ["street_vendor", "Street Vendor / Hawker"], ["artisan", "Artisan / Vishwakarma"], ["unemployed", "Unemployed / Homemaker"], ["entrepreneur", "Entrepreneur / Self-Employed"], ["laborer", "Unorganized Worker / Laborer"]],
        cat: [["", "--"], ["General", "General"], ["OBC", "OBC"], ["SC", "SC"], ["ST", "ST"]],
        land: [["", "--"], ["true", "Yes (Owns Land)"], ["false", "No (Landless)"]],
        stage_collect: "Step-by-Step",
        stage_done: "Complete ✓",
        welcome_badge: "Step 1 / 7",
        assistant_badge: "Assistant"
      },
      hi: {
        gender: [["", "-- चुनें --"], ["male", "पुरुष (Male)"], ["female", "महिला (Female)"], ["other", "अन्य (Other)"]],
        occ: [["", "-- चुनें --"], ["farmer", "किसान / कृषि"], ["student", "छात्र / विद्यार्थी"], ["street_vendor", "रेहड़ी-पटरी विक्रेता"], ["artisan", "कारीगर / विश्वकर्मा"], ["unemployed", "बेरोजगार / गृहिणी"], ["entrepreneur", "उद्यमी / स्वरोजगार"], ["laborer", "श्रमिक / मजदूर"]],
        cat: [["", "-- चुनें --"], ["General", "सामान्य (General)"], ["OBC", "अन्य पिछड़ा वर्ग (OBC)"], ["SC", "अनुसूचित जाति (SC)"], ["ST", "अनुसूचित जनजाति (ST)"]],
        land: [["", "-- चुनें --"], ["true", "हाँ (कृषि भूमि है)"], ["false", "नहीं (भूमिहीन)"]],
        stage_collect: "चरण-दर-चरण",
        stage_done: "पूर्ण ✓",
        welcome_badge: "चरण 1 / 7",
        assistant_badge: "सहायक"
      },
      ta: {
        gender: [["", "-- தேர்ந்தெடு --"], ["male", "ஆண் (Male)"], ["female", "பெண் (Female)"], ["other", "மற்றவை (Other)"]],
        occ: [["", "-- தேர்ந்தெடு --"], ["farmer", "விவசாயி"], ["student", "மாணவர்"], ["street_vendor", "தெரு வியாபாரி"], ["artisan", "கைவினைஞர்"], ["unemployed", "வேலையில்லாதவர் / இல்லத்தரசி"], ["entrepreneur", "தொழில்முனைவோர்"], ["laborer", "தொழிலாளி"]],
        cat: [["", "-- தேர்ந்தெடு --"], ["General", "பொது (General)"], ["OBC", "இதர பிற்படுத்தப்பட்டோர் (OBC)"], ["SC", "பட்டியல் சாதி (SC)"], ["ST", "பழங்குடியினர் (ST)"]],
        land: [["", "-- தேர்ந்தெடு --"], ["true", "ஆம் (நிலம் உள்ளது)"], ["false", "இல்லை (நிலமற்றவர்)"]],
        stage_collect: "படிப்படியாக",
        stage_done: "முடிந்தது ✓",
        welcome_badge: "படி 1 / 7",
        assistant_badge: "உதவியாளர்"
      },
      kn: {
        gender: [["", "-- ಆಯ್ಕೆಮಾಡಿ --"], ["male", "ಪುರುಷ (Male)"], ["female", "ಮಹಿಳೆ (Female)"], ["other", "ಇತರೆ (Other)"]],
        occ: [["", "-- ಆಯ್ಕೆಮಾಡಿ --"], ["farmer", "ರೈತ / ಕೃಷಿ"], ["student", "ವಿದ್ಯಾರ್ಥಿ"], ["street_vendor", "ಬೀದಿ ವ್ಯಾಪಾರಿ"], ["artisan", "ಕುಶಲಕರ್ಮಿ / ವಿಶ್ವಕರ್ಮ"], ["unemployed", "ನಿರುದ್ಯೋಗಿ / ಗೃಹಿಣಿ"], ["entrepreneur", "ಉದ್ಯಮಿ / ಸ್ವಯಂ ಉದ್ಯೋಗಿ"], ["laborer", "ಕಾರ್ಮಿಕ"]],
        cat: [["", "-- ಆಯ್ಕೆಮಾಡಿ --"], ["General", "ಸಾಮಾನ್ಯ (General)"], ["OBC", "ಇತರೆ ಹಿಂದುಳಿದ ವರ್ಗ (OBC)"], ["SC", "ಪರಿಶಿಷ್ಟ ಜಾತಿ (SC)"], ["ST", "ಪರಿಶಿಷ್ಟ ಪಂಗಡ (ST)"]],
        land: [["", "-- ಆಯ್ಕೆಮಾಡಿ --"], ["true", "ಹೌದು (ಭೂಮಿ ಇದೆ)"], ["false", "ಇಲ್ಲ (ಭೂಮಿ ಇಲ್ಲ)"]],
        stage_collect: "ಹಂತ-ಹಂತವಾಗಿ",
        stage_done: "ಪೂರ್ಣಗೊಂಡಿದೆ ✓",
        welcome_badge: "ಹಂತ 1 / 7",
        assistant_badge: "ಸಹಾಯಕ"
      },
      pa: {
        gender: [["", "-- ਚੁਣੋ --"], ["male", "ਪੁਰਸ਼ (Male)"], ["female", "ਮਹਿਲਾ (Female)"], ["other", "ਹੋਰ (Other)"]],
        occ: [["", "-- ਚੁਣੋ --"], ["farmer", "ਕਿਸਾਨ / ਖੇਤੀਬਾੜੀ"], ["student", "ਵਿਦਿਆਰਥੀ"], ["street_vendor", "ਰੇਹੜੀ-ਫੜ੍ਹੀ ਵਾਲਾ"], ["artisan", "ਕਾਰੀਗਰ / ਵਿਸ਼ਵਕਰਮਾ"], ["unemployed", "ਬੇਰੁਜ਼ਗਾਰ / ਘਰੇਲੂ ਔਰਤ"], ["entrepreneur", "ਉੱਦਮੀ / ਸਵੈ-ਰੁਜ਼ਗਾਰ"], ["laborer", "ਮਜ਼ਦੂਰ / ਕਿਰਤੀ"]],
        cat: [["", "-- ਚੁਣੋ --"], ["General", "ਜਨਰਲ (General)"], ["OBC", "ਪੱਛੜੀ ਸ਼੍ਰੇਣੀ (OBC)"], ["SC", "ਅਨੁਸੂਚਿਤ ਜਾਤੀ (SC)"], ["ST", "ਅਨੁਸੂਚਿਤ ਕਬੀਲਾ (ST)"]],
        land: [["", "-- ਚੁਣੋ --"], ["true", "ਹਾਂ (ਜ਼ਮੀਨ ਹੈ)"], ["false", "ਨਹੀਂ (ਜ਼ਮੀਨ ਨਹੀਂ)"]],
        stage_collect: "ਕਦਮ-ਦਰ-ਕਦਮ",
        stage_done: "ਪੂਰਾ ਹੋਇਆ ✓",
        welcome_badge: "ਪੜਾਅ 1 / 7",
        assistant_badge: "ਸਹਾਇਕ"
      },
      bn: {
        gender: [["", "-- নির্বাচন করুন --"], ["male", "পুরুষ (Male)"], ["female", "মহিলা (Female)"], ["other", "অন্যান্য (Other)"]],
        occ: [["", "-- নির্বাচন করুন --"], ["farmer", "কৃষক / কৃষি"], ["student", "ছাত্র / ছাত্রী"], ["street_vendor", "পথ ব্যবসায়ী / হকার"], ["artisan", "কারিগর / বিশ্বকর্মা"], ["unemployed", "বেকার / গৃহিণী"], ["entrepreneur", "উদ্যোক্তা / স্বনির্ভর"], ["laborer", "শ্রমিক / দিনমজুর"]],
        cat: [["", "-- নির্বাচন করুন --"], ["General", "সাধারণ (General)"], ["OBC", "অন্যান্য অনগ্রসর শ্রেণী (OBC)"], ["SC", "তফসিলি জাতি (SC)"], ["ST", "তফসিলি উপজাতি (ST)"]],
        land: [["", "-- নির্বাচন করুন --"], ["true", "হ্যাঁ (জমি আছে)"], ["false", "না (ভূমিহীন)"]],
        stage_collect: "ধাপে ধাপে",
        stage_done: "সম্পূর্ণ ✓",
        welcome_badge: "ধাপ 1 / 7",
        assistant_badge: "সহায়ক"
      }
    };

    function updateSelectOptions(selectId, optionsList) {
      const sel = document.getElementById(selectId);
      const currentVal = sel.value;
      sel.innerHTML = optionsList.map(([val, label]) => `<option value="${val}">${label}</option>`).join("");
      sel.value = currentVal;
    }

    function t(key) {
      const lang = document.getElementById("langSelect").value || "en";
      return (UI_STRINGS[lang] && UI_STRINGS[lang][key]) || UI_STRINGS.en[key] || key;
    }

    function applyScreenLocalization() {
      document.getElementById("ui_unofficial_badge").textContent = t("unofficial_badge");
      document.getElementById("ui_top_banner").innerHTML = t("top_banner");
      document.getElementById("ui_app_title").textContent = t("app_title");
      document.getElementById("ui_app_subtitle").textContent = t("app_subtitle");
      document.getElementById("ui_lang_label").textContent = t("lang_label");
      document.getElementById("ui_low_bw").textContent = t("low_bw");
      document.getElementById("ui_reset_btn").textContent = t("reset_btn");
      document.getElementById("ui_demo_bar_label").textContent = t("demo_bar_label");
      document.getElementById("ui_oos_btn").textContent = t("oos_btn");
      document.getElementById("ui_profile_title").textContent = t("profile_title");
      document.getElementById("ui_profile_desc").textContent = t("profile_desc");
      document.getElementById("ui_lbl_state").textContent = t("lbl_state");
      document.getElementById("ui_lbl_age").textContent = t("lbl_age");
      document.getElementById("ui_lbl_gender").textContent = t("lbl_gender");
      document.getElementById("ui_lbl_occ").textContent = t("lbl_occ");
      document.getElementById("ui_lbl_income").textContent = t("lbl_income");
      document.getElementById("ui_lbl_cat").textContent = t("lbl_cat");
      document.getElementById("ui_lbl_land").textContent = t("lbl_land");
      document.getElementById("ui_update_prof_btn").textContent = t("update_prof_btn");
      document.getElementById("ui_csc_title").textContent = t("csc_title");
      document.getElementById("ui_csc_link").textContent = t("csc_link");
      document.getElementById("ui_chat_title").textContent = t("chat_title");
      document.getElementById("ui_chat_subtitle").textContent = t("chat_subtitle");
      if (!isRecording) {
        document.getElementById("micLabel").textContent = t("mic_speak");
      }
      document.getElementById("ui_upload_btn").textContent = t("upload_btn");
      document.getElementById("chatInput").placeholder = t("chat_placeholder");
      document.getElementById("ui_send_btn").textContent = t("send_btn");
      document.getElementById("ui_schemes_title").textContent = t("schemes_title");
      document.getElementById("ui_schemes_subtitle").textContent = t("schemes_subtitle");
      document.getElementById("ui_tab_elig").textContent = t("tab_elig");
      document.getElementById("ui_tab_rej").textContent = t("tab_rej");
      const lang = document.getElementById("langSelect").value || "en";
      const d = DROPDOWN_OPTIONS[lang] || DROPDOWN_OPTIONS.en;
      updateSelectOptions("prof_gender", d.gender);
      updateSelectOptions("prof_occupation", d.occ);
      updateSelectOptions("prof_category", d.cat);
      updateSelectOptions("prof_land", d.land);
      renderSchemes();
    }

    async function onLanguageChange() {
      stopAllAudioPlayback();
      applyScreenLocalization();
      const hasProfile = document.getElementById("prof_state").value || document.getElementById("prof_age").value || document.getElementById("prof_occupation").value;
      if (hasProfile) {
        await applyManualProfile();
      } else {
        await resetConversation();
      }
    }

    function stopAllAudioPlayback() {
      if (pendingReplyTimeout) {
        clearTimeout(pendingReplyTimeout);
        pendingReplyTimeout = null;
      }
      const player = document.getElementById("audioPlayer");
      if (player) {
        player.onended = null;
        player.pause();
        player.currentTime = 0;
        player.removeAttribute("src");
        player.load();
      }
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
      }
    }

    function appendMessage(role, text, subtext = "", badge = "") {
      const box = document.getElementById("chatBox");
      const wrap = document.createElement("div");
      wrap.className = role === "user" ? "flex justify-end" : "flex justify-start";
      const bubble = document.createElement("div");
      bubble.className = role === "user"
        ? "max-w-[85%] rounded-2xl rounded-tr-sm bg-emerald-600 text-white px-3.5 py-2.5 text-sm shadow"
        : "max-w-[88%] rounded-2xl rounded-tl-sm bg-slate-800 border border-slate-700 text-slate-100 px-3.5 py-2.5 text-sm shadow";

      let html = "";
      if (badge) {
        html += `<div class="text-[10px] font-bold uppercase tracking-wider text-amber-300 mb-1">${badge}</div>`;
      }
      html += `<div class="leading-relaxed">${text}</div>`;
      const curLang = document.getElementById("langSelect").value || "en";
      if (subtext && curLang === "en") {
        html += `<div class="mt-1.5 pt-1.5 border-t border-white/10 text-[11px] opacity-80">${subtext}</div>`;
      }
      bubble.innerHTML = html;
      wrap.appendChild(bubble);
      box.appendChild(wrap);
      box.scrollTop = box.scrollHeight;
    }

    function playReplyAudio(base64Audio, textFallback, langCode) {
      stopAllAudioPlayback();
      if (document.getElementById("lowBwToggle").checked) return;
      const player = document.getElementById("audioPlayer");
      if (base64Audio && base64Audio.length > 50) {
        player.src = `data:audio/mpeg;base64,${base64Audio}`;
        player.play().catch(() => {});
      } else if ('speechSynthesis' in window && textFallback) {
        const u = new SpeechSynthesisUtterance(textFallback);
        u.lang = SPEECH_LANG_MAP[langCode] || "en-IN";
        window.speechSynthesis.speak(u);
      }
    }

    function syncProfileUI(profile, stage, csc) {
      if (!profile) return;
      document.getElementById("prof_state").value = profile.state || "";
      document.getElementById("prof_age").value = profile.age !== undefined && profile.age !== null ? profile.age : "";
      document.getElementById("prof_gender").value = profile.gender || "";
      document.getElementById("prof_occupation").value = profile.occupation || "";
      document.getElementById("prof_income").value = profile.income !== undefined && profile.income !== null ? profile.income : "";
      document.getElementById("prof_category").value = profile.category || "";
      document.getElementById("prof_land").value = profile.owns_land === true ? "true" : (profile.owns_land === false ? "false" : "");

      const badge = document.getElementById("stageBadge");
      if (stage === "results") {
        badge.textContent = "Complete ✓";
        badge.className = "text-[11px] px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30";
      } else if (stage === "out_of_scope_refusal") {
        badge.textContent = "Out-of-Scope Guard";
        badge.className = "text-[11px] px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30";
      } else {
        badge.textContent = "Collecting Info";
        badge.className = "text-[11px] px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30";
      }

      if (csc) {
        document.getElementById("cscName").textContent = csc.center_name;
        document.getElementById("cscInstr").textContent = csc.instructions;
      }
    }

    function switchSchemeTab(tab) {
      activeTab = tab;
      document.getElementById("tabEligibleBtn").className = tab === "eligible"
        ? "px-2.5 py-1 rounded-lg bg-emerald-600 text-white font-semibold"
        : "px-2.5 py-1 rounded-lg text-slate-400 hover:text-white";
      document.getElementById("tabRejectedBtn").className = tab === "rejected"
        ? "px-2.5 py-1 rounded-lg bg-rose-600 text-white font-semibold"
        : "px-2.5 py-1 rounded-lg text-slate-400 hover:text-white";
      renderSchemes();
    }

    function renderSchemes() {
      const container = document.getElementById("schemesContainer");
      document.getElementById("eligCount").textContent = currentMatches.length;
      document.getElementById("rejCount").textContent = currentRejected.length;

      const list = activeTab === "eligible" ? currentMatches : currentRejected;
      if (!list || list.length === 0) {
        container.innerHTML = `<div class="text-center text-slate-400 text-xs py-10">${t("empty_schemes")}</div>`;
        return;
      }

      container.innerHTML = list.map(s => {
        const isElig = s.status === "eligible";
        const isPoss = s.status === "possibly_eligible";
        const badgeClass = isElig
          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
          : (isPoss ? "bg-amber-500/20 text-amber-300 border-amber-500/30" : "bg-rose-500/20 text-rose-300 border-rose-500/30");
        const badgeLabel = isElig ? t("badge_elig") : (isPoss ? t("badge_poss") : t("badge_rej"));

        const reasonsHtml = (s.reasons || []).map(r => `<li class="flex items-start gap-1.5"><span>•</span><span>${r}</span></li>`).join("");
        const docsHtml = (s.documents_required || []).map(d => `<li class="flex items-center gap-1.5"><span>☑️</span><span>${d}</span></li>`).join("");
        const stepsHtml = (s.application_steps || []).map((st, i) => `<li><b>${t("step_word")} ${i+1}:</b> ${st}</li>`).join("");

        return `
          <div class="bg-slate-900/95 border border-slate-800 rounded-xl p-3.5 space-y-2.5 shadow">
            <div class="flex items-start justify-between gap-2">
              <h3 class="text-sm font-bold text-white leading-snug">${s.scheme_name}</h3>
              <span class="text-[10px] font-bold px-2 py-0.5 rounded-full border whitespace-nowrap ${badgeClass}">${badgeLabel}</span>
            </div>
            <p class="text-xs text-slate-300 leading-relaxed">${s.benefits_summary}</p>
            <div class="bg-slate-950/70 rounded-lg p-2.5 text-[11px] text-slate-300 space-y-1">
              <div class="font-semibold text-slate-200">${isElig || isPoss ? t("rule_verif") : t("why_not")}</div>
              <ul class="space-y-0.5 text-slate-400">${reasonsHtml}</ul>
            </div>
            ${isElig || isPoss ? `
              <details class="text-xs bg-slate-950/50 rounded-lg p-2 border border-slate-800/80" open>
                <summary class="font-semibold text-emerald-300 cursor-pointer">${t("checklist_summary")}</summary>
                <div class="mt-2 space-y-2 text-[11px]">
                  <div>
                    <div class="font-bold text-slate-200 mb-1">${t("req_docs")}</div>
                    <ul class="space-y-1 text-slate-300">${docsHtml}</ul>
                  </div>
                  <div>
                    <div class="font-bold text-slate-200 mb-1">${t("how_to_apply")}</div>
                    <ol class="space-y-1 text-slate-300">${stepsHtml}</ol>
                  </div>
                </div>
              </details>
            ` : ""}
            <div class="pt-1 flex items-center justify-between text-[11px] text-slate-400 border-t border-slate-800/80">
              <span>${t("verified_word")} ${s.date_checked}</span>
              <a href="${s.source_url}" target="_blank" class="text-sky-400 hover:underline font-semibold">${t("source_link")}</a>
            </div>
          </div>
        `;
      }).join("");
    }

    async function runBundledSample(sampleFile, langCode) {
      stopAllAudioPlayback();
      document.getElementById("langSelect").value = langCode;
      applyScreenLocalization();
      document.getElementById("voiceStatus").textContent = `Playing & transcribing ${sampleFile}...`;

      const player = document.getElementById("audioPlayer");
      player.src = `/api/samples/${sampleFile}`;
      player.play().catch(() => {});

      const r = await fetch(`/api/voice/sample_turn/${encodeURIComponent(sampleFile)}?session_id=${encodeURIComponent(sessionId)}`, { method: "POST" });
      const data = await r.json();
      sessionId = data.session_id;

      appendMessage("user", `🎙️ [${sampleFile}] ${data.transcript}`, data.normalized_text, `Voice Input (${data.detected_language.toUpperCase()})`);
      appendMessage("assistant", data.reply, data.english_reply !== data.reply ? data.english_reply : "", `Grounded Agent Reply (${data.detected_language.toUpperCase()})`);

      syncProfileUI(data.profile, data.stage, data.csc_guidance);
      currentMatches = data.matches || [];
      currentRejected = data.rejected_matches || [];
      switchSchemeTab("eligible");
      document.getElementById("voiceStatus").textContent = `Completed ${sampleFile} voice pipeline!`;

      player.onended = () => {
        player.onended = null;
        playReplyAudio(data.reply_audio_base64, data.reply, data.detected_language);
      };
    }

    async function uploadAudioFile(input) {
      if (!input.files || !input.files[0]) return;
      stopAllAudioPlayback();
      const file = input.files[0];
      const lang = document.getElementById("langSelect").value;
      document.getElementById("voiceStatus").textContent = `Uploading & transcribing ${file.name}...`;

      const form = new FormData();
      form.append("file", file);
      form.append("language", lang);
      form.append("session_id", sessionId);

      const r = await fetch("/api/voice/turn", { method: "POST", body: form });
      const data = await r.json();
      if (!r.ok) {
        appendMessage("assistant", "⚠️ " + (data.detail || "Audio error"), "", "Notice");
        return;
      }
      sessionId = data.session_id;
      document.getElementById("langSelect").value = data.detected_language || lang;
      applyScreenLocalization();

      appendMessage("user", `🎙️ ${data.transcript}`, data.normalized_text, `Uploaded Audio (${data.detected_language.toUpperCase()})`);
      appendMessage("assistant", data.reply, data.english_reply !== data.reply ? data.english_reply : "", "Assistant");

      syncProfileUI(data.profile, data.stage, data.csc_guidance);
      currentMatches = data.matches || [];
      currentRejected = data.rejected_matches || [];
      renderSchemes();
      playReplyAudio(data.reply_audio_base64, data.reply, data.detected_language);
      document.getElementById("voiceStatus").textContent = "Voice turn complete";
      input.value = "";
    }

    let mediaRecorder = null;
    let recordedBlobs = [];

    async function convertBlobToWav(blob) {
      const arrayBuffer = await blob.arrayBuffer();
      const actx = new (window.AudioContext || window.webkitAudioContext)();
      const audioBuffer = await actx.decodeAudioData(arrayBuffer);
      const targetRate = 16000;
      const offlineCtx = new OfflineAudioContext(1, Math.ceil(audioBuffer.duration * targetRate), targetRate);
      const source = offlineCtx.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(offlineCtx.destination);
      source.start(0);
      const rendered = await offlineCtx.startRendering();
      const samples = rendered.getChannelData(0);
      await actx.close();

      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);
      const writeString = (offset, str) => {
        for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
      };
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, targetRate, true);
      view.setUint32(28, targetRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, samples.length * 2, true);
      let offset = 44;
      for (let i = 0; i < samples.length; i++, offset += 2) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
      }
      return new Blob([view], { type: 'audio/wav' });
    }

    async function toggleMicRecording() {
      const btn = document.getElementById("micBtn");
      const label = document.getElementById("micLabel");
      const statusEl = document.getElementById("voiceStatus");
      const lang = document.getElementById("langSelect").value;

      if (!isRecording) {
        stopAllAudioPlayback();
        browserSpeechText = "";
        document.getElementById("chatInput").value = "";

        // 1. Start undisturbed Chrome/Edge Web Speech API
        const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (SpeechRec) {
          try {
            activeSpeechRec = new SpeechRec();
            activeSpeechRec.lang = SPEECH_LANG_MAP[lang] || "en-IN";
            activeSpeechRec.continuous = true;
            activeSpeechRec.interimResults = true;
            activeSpeechRec.onresult = (ev) => {
              let txt = "";
              for (let i = 0; i < ev.results.length; i++) {
                txt += ev.results[i][0].transcript + " ";
              }
              browserSpeechText = txt.trim();
              if (browserSpeechText) {
                document.getElementById("chatInput").value = browserSpeechText;
                statusEl.textContent = `Heard live: "${browserSpeechText}"`;
              }
            };
            activeSpeechRec.onerror = () => {};
            activeSpeechRec.start();
          } catch (e) {}
        }

        // 2. Clean native MediaRecorder (does NOT open an active AudioContext while recording, preventing mic conflict!)
        try {
          mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true }
          });
          recordedBlobs = [];
          mediaRecorder = new MediaRecorder(mediaStream);
          mediaRecorder.ondataavailable = (ev) => {
            if (ev.data && ev.data.size > 0) recordedBlobs.push(ev.data);
          };
          mediaRecorder.start(200);

          isRecording = true;
          recSeconds = 0;
          btn.classList.add("pulse-ring");
          label.textContent = "⏹ Stop (0s)";
          statusEl.textContent = `🎙️ Listening (${lang.toUpperCase()})... Speak naturally, then click Stop!`;
          recTimerInterval = setInterval(() => {
            recSeconds++;
            label.textContent = `⏹ Stop (${recSeconds}s)`;
          }, 1000);
        } catch (err) {
          alert("Microphone error: " + err.message);
        }
      } else {
        isRecording = false;
        clearInterval(recTimerInterval);
        btn.classList.remove("pulse-ring");
        label.textContent = t("mic_speak");

        if (activeSpeechRec) {
          try { activeSpeechRec.stop(); } catch (e) {}
          activeSpeechRec = null;
        }
        if (mediaRecorder && mediaRecorder.state !== "inactive") {
          await new Promise(resolve => {
            mediaRecorder.onstop = resolve;
            mediaRecorder.stop();
          });
        }
        if (mediaStream) {
          mediaStream.getTracks().forEach(tr => tr.stop());
        }

        statusEl.textContent = "Processing your speech...";
        await new Promise(r => setTimeout(r, 300));

        // If browser live speech recognition caught your words, send immediately!
        const capturedText = (browserSpeechText || document.getElementById("chatInput").value || "").trim();
        if (capturedText) {
          document.getElementById("chatInput").value = capturedText;
          await sendTextTurn(true);
          statusEl.textContent = "Voice turn complete!";
          return;
        }

        // Otherwise convert recorded audio with high-quality OfflineAudioContext anti-aliased resampling to 16kHz WAV
        try {
          const rawBlob = new Blob(recordedBlobs, { type: mediaRecorder.mimeType || "audio/webm" });
          const wavBlob = await convertBlobToWav(rawBlob);
          const file = new File([wavBlob], "mic_recording.wav", { type: "audio/wav" });
          const form = new FormData();
          form.append("file", file);
          form.append("language", lang);
          form.append("session_id", sessionId);

          const r = await fetch("/api/voice/turn", { method: "POST", body: form });
          const data = await r.json();
          if (!r.ok) {
            statusEl.textContent = data.detail || "Could not hear clearly.";
            appendMessage("assistant", "⚠️ " + (data.detail || "Could not hear clearly. Please try speaking again."), "", "Voice Notice");
            return;
          }
          sessionId = data.session_id;
          appendMessage("user", `🎙️ ${data.transcript}`, data.normalized_text !== data.transcript ? data.normalized_text : "", `Live Mic (${data.detected_language.toUpperCase()})`);
          appendMessage("assistant", data.reply, data.english_reply !== data.reply ? data.english_reply : "", "Grounded Assistant");
          syncProfileUI(data.profile, data.stage, data.csc_guidance);
          currentMatches = data.matches || [];
          currentRejected = data.rejected_matches || [];
          renderSchemes();
          playReplyAudio(data.reply_audio_base64, data.reply, data.detected_language);
          statusEl.textContent = "Voice turn complete!";
        } catch (err) {
          statusEl.textContent = "Could not process audio — please try again.";
        }
      }
    }

    async function sendTextTurn(fromVoice = false) {
      stopAllAudioPlayback();
      const input = document.getElementById("chatInput");
      const msg = input.value.trim();
      if (!msg) return;
      input.value = "";
      const lang = document.getElementById("langSelect").value;

      appendMessage("user", fromVoice ? `🎙️ ${msg}` : msg, "", fromVoice ? `Live Voice (${lang.toUpperCase()})` : `User (${lang.toUpperCase()})`);

      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId || null, message: msg, language: lang })
      });
      const data = await r.json();
      sessionId = data.session_id;

      appendMessage("assistant", data.reply, data.english_reply !== data.reply ? data.english_reply : "", "Grounded Assistant");
      syncProfileUI(data.profile, data.stage, data.csc_guidance);
      currentMatches = data.matches || [];
      currentRejected = data.rejected_matches || [];
      renderSchemes();
      playReplyAudio(data.reply_audio_base64, data.reply, lang);
    }

    async function applyManualProfile() {
      stopAllAudioPlayback();
      const lang = document.getElementById("langSelect").value;
      const profile = {
        state: document.getElementById("prof_state").value || null,
        age: document.getElementById("prof_age").value ? parseInt(document.getElementById("prof_age").value) : null,
        gender: document.getElementById("prof_gender").value || null,
        occupation: document.getElementById("prof_occupation").value || null,
        income: document.getElementById("prof_income").value ? parseInt(document.getElementById("prof_income").value) : null,
        category: document.getElementById("prof_category").value || null,
        owns_land: document.getElementById("prof_land").value === "true" ? true : (document.getElementById("prof_land").value === "false" ? false : null)
      };
      const r = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId || null, message: "Updated profile details", language: lang, profile })
      });
      const data = await r.json();
      sessionId = data.session_id;
      appendMessage("assistant", data.reply, data.english_reply !== data.reply ? data.english_reply : "", "Profile Updated");
      syncProfileUI(data.profile, data.stage, data.csc_guidance);
      currentMatches = data.matches || [];
      currentRejected = data.rejected_matches || [];
      renderSchemes();
      playReplyAudio(data.reply_audio_base64, data.reply, lang);
    }

    async function runPersonaPreset(type) {
      stopAllAudioPlayback();
      if (type === "bengali_woman") {
        document.getElementById("langSelect").value = "bn";
        applyScreenLocalization();
        document.getElementById("chatInput").value = "আমি পশ্চিমবঙ্গের ৩৪ বছর বয়সী মহিলা, গৃহিণী, বার্ষিক আয় ১১০০০০ টাকা, SC বিভাগ, জমি নেই।";
        await sendTextTurn(true);
      }
    }

    async function testOutOfScope() {
      stopAllAudioPlayback();
      document.getElementById("chatInput").value = "How do I apply for a US tourist visa and buy Bitcoin?";
      await sendTextTurn(false);
    }

    async function resetConversation() {
      // Immediately cut off all playing audio and speech synthesis!
      stopAllAudioPlayback();
      if (isRecording) {
        isRecording = false;
        clearInterval(recTimerInterval);
        document.getElementById("micBtn").classList.remove("pulse-ring");
      }
      applyScreenLocalization();

      const lang = document.getElementById("langSelect").value;
      const r = await fetch(`/api/session?language=${lang}`, { method: "POST" });
      const data = await r.json();
      sessionId = data.session_id;
      document.getElementById("chatBox").innerHTML = "";
      currentMatches = [];
      currentRejected = [];
      syncProfileUI({}, "collecting_profile", null);
      renderSchemes();
      const d = DROPDOWN_OPTIONS[lang] || DROPDOWN_OPTIONS.en;
      appendMessage("assistant", data.reply, "", d.welcome_badge);
      document.getElementById("voiceStatus").textContent = d.stage_collect;
    }

    window.onload = () => {
      applyScreenLocalization();
      resetConversation();
    };
  </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    return HTMLResponse(content=FRONTEND_HTML)
