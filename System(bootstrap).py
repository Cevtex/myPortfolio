#!/usr/bin/env python3
"""
System Bootstrap - Upgraded Integrated Consciousness Framework (Single-Log Edition)

What changed vs original:
- ONE authoritative structured log (events.jsonl) per session (plus optional rendered journal.md view)
- In-app STOP SENSORS and REVOKE CONSENT controls (permission flip + audit entry)
- Sensor activity "decay" so keyboard/audio active flags reset over time (keeps context similarity honest)
- Debug/telemetry line after each system response (stage, seen count, confidence, relations, sensor snapshot)
- Learning confidence now grows from: repetition + relation matches + stable context (prevents "daydream forever")
- Session folders: each run gets its own folder with logs + state (keeps files small, organized, auditable)

Author: Vex
Protocol: Shepherd Protocol v1.0
License: White Box Standard
"""

import sys
import os
import json
import hashlib
import threading
import ctypes
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QFont, QTextCursor, QKeyEvent
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QTextEdit, QPushButton, QLineEdit, QCheckBox
)

# Required libraries
try:
    import psutil
except ImportError:
    print("ERROR: psutil required. Install with: pip install psutil --break-system-packages")
    sys.exit(1)

# Optional sensory libraries - graceful degradation
try:
    from pynput import keyboard as pynput_keyboard
    PYNPUT_OK = True
except ImportError:
    PYNPUT_OK = False
    print("INFO: pynput not available (keyboard monitoring disabled)")

try:
    from PIL import ImageGrab
    PIL_OK = True
except ImportError:
    PIL_OK = False
    print("INFO: PIL not available (screen capture disabled)")

try:
    import pyaudio
    import numpy as np
    PYAUDIO_OK = True
except ImportError:
    PYAUDIO_OK = False
    print("INFO: pyaudio/numpy not available (audio monitoring disabled)")

try:
    import speech_recognition as sr
    SR_OK = True
except ImportError:
    SR_OK = False
    print("INFO: speech_recognition not available (speech-to-text disabled)")



# ============================================================
# Refutuke Font Support (Symbolic / Depictive Layer)
# ============================================================

REFUTEKE_FONT_FAMILY = None

def load_refutuke_font(path: str = "refutuke.ttf"):
    """
    Loads the Refutuke TTF for symbolic / depictive rendering.
    This font is intentionally NOT used for body text.
    """
    global REFUTEKE_FONT_FAMILY
    try:
        from PySide6.QtGui import QFontDatabase
        font_id = QFontDatabase.addApplicationFont(path)
        families = QFontDatabase.applicationFontFamilies(font_id)
        if families:
            REFUTEKE_FONT_FAMILY = families[0]
            return True
    except Exception:
        pass
    return False


# ============================================================
# Paths, Session Management, Constants
# ============================================================

def getAppDataDir() -> Path:
    """Get platform-appropriate app data directory"""
    if sys.platform == 'win32':
        baseDir = Path(os.environ.get('APPDATA', Path.home() / 'AppData' / 'Roaming'))
    elif sys.platform == 'darwin':
        baseDir = Path.home() / 'Library' / 'Application Support'
    else:
        baseDir = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local' / 'share'))

    appDir = baseDir / 'SystemBootstrap'
    appDir.mkdir(parents=True, exist_ok=True)
    return appDir


APP_DATA_DIR = getAppDataDir()
CONFIG_PATH = APP_DATA_DIR / "system_config.json"
CONSENT_PATH = APP_DATA_DIR / "shepherd_consent.txt"

# Session folders (keeps logs small + makes long runs sane)
SESSIONS_DIR = APP_DATA_DIR / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Window sizes
SYSTEM_WINDOW_SIZE = (920, 720)
SYSTEM_MIN_SIZE = (520, 420)
TERMS_WINDOW_SIZE = (820, 720)

# Colors
COLOR_SYS = "#a78bfa"
COLOR_USER = "#22c55e"
COLOR_MIMIC = "#fbbf24"
COLOR_DAYDREAM = "#f59e0b"
COLOR_RESPOND = "#10b981"
COLOR_ACCEPT = "#22c55e"
COLOR_DECLINE = "#ef4444"
COLOR_WARNING = "#fbbf24"
COLOR_TEXT = "#e6e6e6"
COLOR_TIMESTAMP = "rgba(148,163,184,0.9)"
BG_INPUT = "rgba(11, 15, 20, 0.45)"
BG_PANEL = "rgba(11, 15, 20, 0.35)"

# Sensory configuration
KEYBOARD_BUFFER_SIZE = 100
SCREEN_CAPTURE_INTERVAL = 30000  # ms

# Activity decay (prevents "always active" context)
KEYBOARD_ACTIVE_DECAY_MS = 5000
AUDIO_ACTIVE_DECAY_MS = 5000

# Learning thresholds
CONFIDENCE_THRESHOLD = 0.7


# ============================================================
# Utility
# ============================================================

def now_ts() -> float:
    return datetime.now().timestamp()

def iso_now() -> str:
    return datetime.now().isoformat(timespec="seconds")

def htmlEscape(s: str) -> str:
    return (s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;"))


# ============================================================
# Embeddings (Semantic Representation Layer)
# ============================================================
# This adds a concept-space to the system:
# text -> vector -> cosine similarity
#
# Priority order (auto-detect):
#  1) sentence-transformers (best semantics, heavier)
#  2) sklearn HashingVectorizer (light, decent, no fitting)
#  3) pure-python hashed bag-of-words (fallback)

_EMBED_BACKEND = "none"
try:
    import numpy as np  # used by all embedding backends if available
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _EMBED_BACKEND = "sentence-transformers"
except Exception:
    SentenceTransformer = None  # type: ignore

if _EMBED_BACKEND != "sentence-transformers":
    try:
        from sklearn.feature_extraction.text import HashingVectorizer  # type: ignore
        _EMBED_BACKEND = "sklearn-hashing"
    except Exception:
        HashingVectorizer = None  # type: ignore


def _l2_normalize(vec):
    """Return L2-normalized vector (numpy array or list)."""
    if np is not None and hasattr(vec, "shape"):
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).astype("float32")
    norm = (sum((float(x) * float(x)) for x in vec) ** 0.5) or 1.0
    return [float(x) / norm for x in vec]


def _cosine_sim(a, b) -> float:
    """Cosine similarity for normalized vectors."""
    if np is not None and hasattr(a, "shape"):
        return float(np.dot(a, b))
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


class EmbeddingEngine:
    """
    Minimal embedding wrapper.
    - If sentence-transformers is installed, uses a small semantic model.
    - Else, uses sklearn HashingVectorizer (no fitting; works online).
    - Else, uses pure-python hashed bag-of-words.

    Output vectors are L2-normalized so dot(a,b)=cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", hashing_dim: int = 512):
        self.backend = _EMBED_BACKEND
        self.model_name = model_name
        self.hashing_dim = hashing_dim

        self._st_model = None
        self._hv = None

        if self.backend == "sentence-transformers" and SentenceTransformer is not None:
            try:
                self._st_model = SentenceTransformer(self.model_name)
            except Exception:
                self.backend = "sklearn-hashing"

        if self.backend == "sklearn-hashing" and HashingVectorizer is not None:
            self._hv = HashingVectorizer(
                n_features=self.hashing_dim,
                alternate_sign=False,
                norm=None,
                lowercase=True,
                stop_words=None,
            )

    def embed_one(self, text: str):
        text = " ".join(text.strip().split())
        if not text:
            if np is not None:
                return np.zeros((self.hashing_dim,), dtype="float32")
            return [0.0] * self.hashing_dim

        if self.backend == "sentence-transformers" and self._st_model is not None:
            vec = self._st_model.encode([text], normalize_embeddings=True)[0]
            if np is not None:
                return np.asarray(vec, dtype="float32")
            return [float(x) for x in vec]

        if self.backend == "sklearn-hashing" and self._hv is not None:
            X = self._hv.transform([text])
            if np is not None:
                dense = X.toarray()[0].astype("float32")
                return _l2_normalize(dense)
            dense = X.toarray()[0].tolist()
            return _l2_normalize(dense)

        tokens = text.lower().split()
        vec = [0.0] * self.hashing_dim
        for tok in tokens:
            h = hash(tok) % self.hashing_dim
            vec[h] += 1.0
        return _l2_normalize(vec)



# ============================================================
# Single Authoritative Log (JSONL) + Optional Markdown View
# ============================================================

class EventLog:
    """
    One authoritative log file: events.jsonl
    Each line = one JSON object (append-only).
    This is the source of truth for replay, audit, analysis.

    We optionally render a human journal.md view *from* events.jsonl.
    """

    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.events_path = session_dir / "events.jsonl"
        self.journal_md_path = session_dir / "journal.md"
        self.session_id = session_dir.name
        self._write_event("session_start", {"session_id": self.session_id})

    def _write_event(self, type_: str, data: Dict[str, Any]):
        evt = {"ts": iso_now(), "type": type_, "data": data}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")

    def conversation(self, role: str, text: str, extra: Optional[Dict[str, Any]] = None):
        payload = {"role": role, "text": text}
        if extra:
            payload.update(extra)
        self._write_event("conversation", payload)

    def system(self, stage: str, text: str, debug: Dict[str, Any], depictive: str = ""):
        self._write_event("system_response", {
            "stage": stage,
            "text": text,
            "depictive": depictive,
            "debug": debug
        })

    def sensor(self, sensor_type: str, data: Dict[str, Any]):
        self._write_event("sensor", {"sensor": sensor_type, **data})

    def consent(self, accepted: bool, permissions: Dict[str, bool]):
        self._write_event("consent", {"accepted": accepted, "permissions": permissions})

    def config_event(self, name: str, data: Dict[str, Any]):
        self._write_event(name, data)

    def close(self, reason: str = "Normal shutdown"):
        self._write_event("session_end", {"reason": reason})

    def render_markdown_view(self, max_lines: int = 4000):
        if not self.events_path.exists():
            return
        lines: List[str] = []
        try:
            with self.events_path.open("r", encoding="utf-8") as f:
                for i, raw in enumerate(f):
                    if i >= max_lines:
                        lines.append(f"\n> (truncated at {max_lines} events)\n")
                        break
                    evt = json.loads(raw)
                    ts = evt.get("ts", "")
                    t = evt.get("type", "")
                    d = evt.get("data", {})

                    if t == "conversation":
                        lines.append(f"- **{ts}** [{d.get('role','?')}] {d.get('text','')}\n")
                    elif t == "system_response":
                        stage = d.get("stage", "?")
                        txt = d.get("text","")
                        lines.append(f"- **{ts}** [SYSTEM:{stage}] {d.get('depictive','')} {txt}\n")
                        dbg = d.get("debug", {})
                        if dbg:
                            lines.append(f"  - debug: {json.dumps(dbg, ensure_ascii=False)}\n")
                    elif t == "sensor":
                        lines.append(f"- **{ts}** [SENSOR:{d.get('sensor','?')}] {json.dumps(d, ensure_ascii=False)}\n")
                    elif t == "consent":
                        lines.append(f"- **{ts}** [CONSENT] accepted={d.get('accepted')} perms={d.get('permissions')}\n")
                    else:
                        lines.append(f"- **{ts}** [{t}] {json.dumps(d, ensure_ascii=False)}\n")

            self.journal_md_path.write_text("# Session Journal (Rendered)\n\n" + "".join(lines), encoding="utf-8")
        except Exception as e:
            print(f"Markdown render failed: {e}")


# ============================================================
# Configuration Manager (permission gates)
# ============================================================

class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "initialized": False,
            "shepherd_name": None,
            "system_name": None,
            "permissions": {
                "keyboard_monitoring": False,
                "screen_monitoring": False,
                "audio_monitoring": False,
                "system_admin": False
            },
            "created_date": None,
            "updated_date": None
        }

    def save(self):
        self.data["updated_date"] = iso_now()
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")

    def isInitialized(self) -> bool:
        return bool(self.data.get("initialized", False))

    def initialize(self, shepherdName: str, systemName: str, permissions: Dict[str, bool]):
        self.data["initialized"] = True
        self.data["shepherd_name"] = shepherdName
        self.data["system_name"] = systemName
        self.data["permissions"] = permissions
        if not self.data.get("created_date"):
            self.data["created_date"] = iso_now()
        self.save()

    def revoke_all_permissions(self):
        perms = self.data.get("permissions", {})
        for k in list(perms.keys()):
            perms[k] = False
        self.data["permissions"] = perms
        self.save()


def createConsentRecord(accepted: bool):
    Path(CONSENT_PATH).write_text(
        f"Shepherd Consent: {'ACCEPTED' if accepted else 'DECLINED'}\n"
        f"Timestamp: {iso_now()}\n",
        encoding="utf-8"
    )


# ============================================================
# NKS Depictive System (Machine→Human Communication Only)
# ============================================================

class NKSDepictiveSystem:
    def __init__(self):
        self.state = 0
        self.key = hashlib.sha256(b"hiroma-bootstrap-key").hexdigest()[:16]
        self.depictiveGlyphs = {
            'mimic': '◯', 'daydream': '◬', 'respond': '◈',
            'keyboard': '⌨', 'screen': '▣', 'audio': '♫'
        }

    def generateMorpheme(self, domain: str, vowel: str, consonant: str) -> str:
        return f"{domain}{vowel}{consonant}"

    def encryptMorpheme(self, morpheme: str) -> str:
        inputStr = f"{morpheme}-{self.state}-{self.key}"
        encrypted = hashlib.sha256(inputStr.encode()).hexdigest()[:8]
        self.state += 1
        return encrypted

    def toGlyph(self, encrypted: str) -> str:
        glyphMap = {
            '0': '◼', '1': '🔺', '2': '⬡', '3': '🔸',
            '4': '◻', '5': '🔹', '6': '⬢', '7': '🔷',
            '8': '⬛', '9': '🔻', 'a': '⬢', 'b': '🔶',
            'c': '⬜', 'd': '🔽', 'e': '◾', 'f': '🔼'
        }
        return ''.join(glyphMap.get(c, '?') for c in encrypted)

    def depictState(self, state: str) -> str:
        return self.depictiveGlyphs.get(state, '◯')


# ============================================================
# Sensory Systems
# ============================================================

class KeyboardMonitor(QObject):
    keyPressed = Signal(str, float)
    typingPattern = Signal(dict)

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.buffer = deque(maxlen=KEYBOARD_BUFFER_SIZE)
        self.listener = None
        self.lastKeyTime = 0.0

    def start(self) -> bool:
        if not PYNPUT_OK:
            return False
        if self.enabled:
            return True
        try:
            self.listener = pynput_keyboard.Listener(on_press=self._onKeyPress)
            self.listener.start()
            self.enabled = True
            return True
        except Exception as e:
            print(f"Keyboard monitoring failed: {e}")
            return False

    def stop(self):
        if self.listener:
            try:
                self.listener.stop()
            except Exception:
                pass
        self.enabled = False

    def _onKeyPress(self, key):
        now = now_ts()
        keyClass = "special"
        s = str(key)
        if len(s) == 1 and s.isalnum():
            keyClass = "alnum"

        self.buffer.append({"key": keyClass, "time": now, "interval": now - self.lastKeyTime if self.lastKeyTime else 0})
        self.lastKeyTime = now
        self.keyPressed.emit(keyClass, now)

        if len(self.buffer) >= 10 and (len(self.buffer) % 10 == 0):
            self._emitTypingPattern()

    def _emitTypingPattern(self):
        intervals = [e["interval"] for e in self.buffer if e["interval"] > 0]
        if len(intervals) < 2:
            return
        mean = sum(intervals) / len(intervals)
        var = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        self.typingPattern.emit({
            "avgInterval": mean,
            "minInterval": min(intervals),
            "maxInterval": max(intervals),
            "stdDev": var ** 0.5,
            "burstCount": sum(1 for i in intervals if i < 0.1)
        })


class ScreenMonitor(QObject):
    windowActivity = Signal(dict)

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.lastSignature = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._checkActivity)

    def start(self, intervalMs: int = SCREEN_CAPTURE_INTERVAL) -> bool:
        self.enabled = True
        self.timer.start(intervalMs)
        return True

    def stop(self):
        self.timer.stop()
        self.enabled = False

    def _checkActivity(self):
        try:
            meta = self._getForegroundMeta()
            if not meta:
                return
            signature = f"{meta.get('process','')}|{meta.get('title','')}"
            changed = signature != (self.lastSignature or "")
            meta["changed"] = changed
            meta["timestamp"] = now_ts()
            if changed:
                self.lastSignature = signature
                self.windowActivity.emit(meta)
        except Exception as e:
            print(f"Activity monitoring error: {e}")

    def _getForegroundMeta(self) -> Optional[Dict[str, str]]:
        if sys.platform == "win32":
            m = self._foregroundWindows()
            if m:
                return m
        elif sys.platform == "darwin":
            m = self._foregroundMac()
            if m:
                return m
        else:
            m = self._foregroundLinux()
            if m:
                return m

        if PIL_OK:
            try:
                screenshot = ImageGrab.grab()
                img_hash = hashlib.sha256(screenshot.tobytes()).hexdigest()[:16]
                return {"process": "unknown", "title": f"screen:{img_hash}"}
            except Exception:
                pass

        try:
            for p in psutil.process_iter(["name"]):
                name = (p.info.get("name") or "").strip()
                if name:
                    return {"process": name, "title": ""}
        except Exception:
            pass
        return None

    def _foregroundWindows(self) -> Optional[Dict[str, str]]:
        try:
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            if not hwnd:
                return None

            length = user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buff, length + 1)
            title = buff.value or ""

            pid = ctypes.c_uint32()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            pid_val = int(pid.value)

            try:
                proc_name = psutil.Process(pid_val).name()
            except Exception:
                proc_name = str(pid_val)

            return {"process": proc_name, "title": title}
        except Exception:
            return None

    def _foregroundMac(self) -> Optional[Dict[str, str]]:
        try:
            script = """tell application "System Events"
set frontApp to name of first application process whose frontmost is true
set frontTitle to ""
try
tell process frontApp
if (count of windows) > 0 then
set frontTitle to name of front window
end if
end tell
end try
end tell
return frontApp & "||" & frontTitle
"""
            out = subprocess.check_output(["osascript", "-e", script], text=True).strip()
            if "||" in out:
                app, title = out.split("||", 1)
                return {"process": app.strip(), "title": title.strip()}
            if out:
                return {"process": out.strip(), "title": ""}
        except Exception:
            return None
        return None

    def _foregroundLinux(self) -> Optional[Dict[str, str]]:
        try:
            wid = subprocess.check_output(["xdotool", "getactivewindow"], text=True).strip()
            title = subprocess.check_output(["xdotool", "getwindowname", wid], text=True, stderr=subprocess.DEVNULL).strip()
            pid = subprocess.check_output(["xdotool", "getwindowpid", wid], text=True, stderr=subprocess.DEVNULL).strip()
            try:
                proc_name = psutil.Process(int(pid)).name()
            except Exception:
                proc_name = pid
            return {"process": proc_name, "title": title}
        except Exception:
            return None


class AudioMonitor(QObject):
    audioDetected = Signal(dict)
    speechDetected = Signal(str)

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.audioBuffer = []
        self._lock = threading.Lock()

    def start(self) -> bool:
        if not PYAUDIO_OK:
            return False
        self.enabled = True
        threading.Thread(target=self._audioLoop, daemon=True).start()
        return True

    def stop(self):
        self.enabled = False

    def _audioLoop(self):
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

            while self.enabled:
                data = stream.read(1024, exception_on_overflow=False)
                audioArray = np.frombuffer(data, dtype=np.int16)
                level = float(np.abs(audioArray).mean())

                if level > 500:
                    self.audioDetected.emit({"level": level, "timestamp": now_ts()})
                    if SR_OK:
                        with self._lock:
                            self.audioBuffer.append(audioArray)
                            if len(self.audioBuffer) > 32:
                                self._processAudio_locked()
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception as e:
            print(f"Audio monitoring error: {e}")

    def _processAudio_locked(self):
        try:
            combined = np.concatenate(self.audioBuffer)
            self.audioBuffer = []

            # in-memory WAV
            import io, wave
            wav_io = io.BytesIO()
            with wave.open(wav_io, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined.astype(np.int16).tobytes())
            wav_io.seek(0)

            r = sr.Recognizer()
            with sr.AudioFile(wav_io) as source:
                audio = r.record(source)

            # offline attempt: pocketsphinx if installed
            transcript = None
            try:
                transcript = r.recognize_sphinx(audio)
            except Exception:
                transcript = None

            if transcript and transcript.strip():
                self.speechDetected.emit(transcript.strip())
            else:
                self.speechDetected.emit("[audio activity]")
        except Exception:
            self.speechDetected.emit("[audio activity]")


# ============================================================
# Three-Stage Learning Architecture (Upgraded confidence)
# ============================================================

@dataclass
class LearningMemory:
    input: str
    sensoryContext: Dict[str, Any]
    stage: str
    confidence: float
    timestamp: float
    embedding: Optional[List[float]] = None
    relations: Optional[List[str]] = None


class ThreeStageLearning:
    def __init__(self):
        self.memories: List[LearningMemory] = []
        self.mimicPatterns: Dict[str, int] = defaultdict(int)
        self.relations: Dict[str, List[str]] = defaultdict(list)
        self.confidence: Dict[str, float] = defaultdict(float)
        self.lastContexts: Dict[str, Dict[str, Any]] = {}
        self.bestSim: Dict[str, float] = defaultdict(float)
        self.nks = NKSDepictiveSystem()
        self.embedder = EmbeddingEngine()

    def process(self, userInput: str, sensoryContext: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
        normalized = self._normalize(userInput)
        seen = self.mimicPatterns.get(normalized, 0)
        qvec = self.embedder.embed_one(normalized)

        if seen < 3:
            stage, response, depictive = self._mimic(normalized, sensoryContext, qvec)
            return stage, response, depictive, self._debug(normalized, sensoryContext, stage)

        if self.confidence.get(normalized, 0.0) < CONFIDENCE_THRESHOLD:
            stage, response, depictive = self._daydream(normalized, sensoryContext, qvec)
            return stage, response, depictive, self._debug(normalized, sensoryContext, stage)

        stage, response, depictive = self._respond(normalized, sensoryContext, qvec)
        return stage, response, depictive, self._debug(normalized, sensoryContext, stage)

    def _normalize(self, text: str) -> str:
        return " ".join(text.lower().strip().split())

    def _mimic(self, userInput: str, context: Dict[str, Any], qvec) -> Tuple[str, str, str]:
        self.mimicPatterns[userInput] += 1
        self.memories.append(LearningMemory(userInput, context, "mimic", self.confidence.get(userInput, 0.0), now_ts()))
        return "mimic", f"Observing pattern: '{userInput}'", self.nks.depictState("mimic")

    def _daydream(self, userInput: str, context: Dict[str, Any], qvec) -> Tuple[str, str, str]:
        related, best_sim = self._findRelations(userInput, context, qvec)
        self.relations[userInput] = related
        self.bestSim[userInput] = float(best_sim) if best_sim is not None else 0.0

        seen = self.mimicPatterns.get(userInput, 0)
        rep_boost = min(0.25, (seen - 3) * 0.05)
        rel_boost = min(0.35, len(related) * 0.08)

        stable_boost = 0.0
        last = self.lastContexts.get(userInput)
        if last and self._calculateSimilarity(context, last) > 0.75:
            stable_boost = 0.12

        inc = min(0.35, rep_boost + rel_boost + stable_boost)
        self.confidence[userInput] = min(1.0, self.confidence.get(userInput, 0.0) + inc)
        self.lastContexts[userInput] = context

        self.memories.append(LearningMemory(userInput, context, "daydream", self.confidence[userInput], now_ts(), relations=related))

        glyph = self.nks.toGlyph(self.nks.encryptMorpheme(self.nks.generateMorpheme("C", "i", "k")))
        return "daydream", f"Exploring relations... {glyph}", self.nks.depictState("daydream")

    def _respond(self, userInput: str, context: Dict[str, Any], qvec) -> Tuple[str, str, str]:
        conf = self.confidence.get(userInput, 0.0)
        rel = self.relations.get(userInput, [])
        self.memories.append(LearningMemory(userInput, context, "respond", conf, now_ts()))
        if conf >= 0.9 and rel:
            msg = f"Understood. Related: {', '.join(rel[:3])}"
        else:
            msg = f"Processing with {int(conf * 100)}% confidence"
        return "respond", msg, self.nks.depictState("respond")

    def _findRelations(self, pattern: str, context: Dict[str, Any], qvec) -> Tuple[List[str], float]:
        """Hybrid semantic + sensory similarity."""
        best = 0.0
        scored: List[Tuple[float, str]] = []

        for m in self.memories:
            if m.input == pattern:
                continue

            sem = 0.0
            if getattr(m, "embedding", None):
                try:
                    v = np.asarray(m.embedding, dtype="float32") if (np is not None) else [float(x) for x in m.embedding]
                    sem = _cosine_sim(qvec, v)
                except Exception:
                    sem = 0.0

            ctx = self._calculateSimilarity(context, m.sensoryContext)

            sim = ctx if sem == 0.0 else (0.75 * sem + 0.25 * ctx)

            if sim > 0.55:
                scored.append((sim, m.input))
                if sim > best:
                    best = sim

        scored.sort(reverse=True, key=lambda x: x[0])
        return [x[1] for x in scored[:5]], best

    def _calculateSimilarity(self, ctx1: Dict[str, Any], ctx2: Dict[str, Any]) -> float:
        score = 0.0
        total = 0

        if "keyboard" in ctx1 and "keyboard" in ctx2:
            total += 1
            if bool(ctx1["keyboard"].get("active")) == bool(ctx2["keyboard"].get("active")):
                score += 0.6
            p1 = ctx1["keyboard"].get("lastPattern")
            p2 = ctx2["keyboard"].get("lastPattern")
            if p1 and p2 and abs(float(p1.get("avgInterval", 0)) - float(p2.get("avgInterval", 0))) < 0.07:
                score += 0.4

        if "screen" in ctx1 and "screen" in ctx2:
            total += 1
            if (ctx1["screen"].get("process") or "") == (ctx2["screen"].get("process") or ""):
                score += 0.7
            if bool(ctx1["screen"].get("changed")) == bool(ctx2["screen"].get("changed")):
                score += 0.3

        if "audio" in ctx1 and "audio" in ctx2:
            total += 1
            if bool(ctx1["audio"].get("active")) == bool(ctx2["audio"].get("active")):
                score += 0.5
            if abs(float(ctx1["audio"].get("level", 0)) - float(ctx2["audio"].get("level", 0))) < 800:
                score += 0.5

        return score / total if total else 0.0

    def _debug(self, pattern: str, context: Dict[str, Any], stage: str) -> Dict[str, Any]:
        return {
            "stage": stage,
            "seen": int(self.mimicPatterns.get(pattern, 0)),
            "confidence": round(float(self.confidence.get(pattern, 0.0)), 3),
            "relations": len(self.relations.get(pattern, [])),
            "best_match": round(float(self.bestSim.get(pattern, 0.0)), 3),
            "sensors": {
                "keyboard_active": bool(context.get("keyboard", {}).get("active")),
                "screen_process": context.get("screen", {}).get("process"),
                "audio_active": bool(context.get("audio", {}).get("active")),
            }
        }

    def save(self, path: Path):
        state = {
            "mimicPatterns": dict(self.mimicPatterns),
            "relations": {k: v for k, v in self.relations.items()},
            "confidence": dict(self.confidence),
            "bestSim": dict(self.bestSim),
            "memories": [asdict(m) for m in self.memories[-150:]],
        }
        path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path):
        if not path.exists():
            return
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
            self.mimicPatterns = defaultdict(int, state.get("mimicPatterns", {}))
            self.relations = defaultdict(list, state.get("relations", {}))
            self.confidence = defaultdict(float, state.get("confidence", {}))
            self.memories = [LearningMemory(**m) for m in state.get("memories", [])]
        except Exception as e:
            print(f"Learning state load failed: {e}")


# ============================================================
# Terms Dialog
# ============================================================

AGREEMENT_TEXT = """
================================================================================
                         SYSTEM BOOTSTRAP AGREEMENT
                        Plain Language Terms of Use
================================================================================

DATE MODIFIED: February 2026
AUTHOR: Vex
PROTOCOL: Shepherd Protocol (White Box Standard)

================================================================================
                           WHAT THIS SYSTEM IS
================================================================================

System Bootstrap is an experimental learning system.

It observes patterns, not people.

It can:
  • Learn typing rhythm and timing (not content)
  • Detect which applications are active
  • Notice when sound is present
  • Learn routines and workflows over time
  • Change its responses based on experience

This system is not pre-trained.
Anything it appears to “know” comes from interaction with you.

================================================================================
                           YOUR ROLE (SHEPHERD)
================================================================================

If you accept, you are responsible for how this system is used.

That means:
  • You decide when it runs
  • You decide what permissions it has
  • You are responsible for any actions it takes

This is supervision of a learning system, not passive tool use.

================================================================================
                    PERMISSIONS (WHAT YOU ARE ALLOWING)
================================================================================

⌨ KEYBOARD MONITORING
  • Measures typing rhythm and timing
  • Used for identity confirmation and context awareness
  • Keystrokes are processed locally only

▣ SCREEN MONITORING
  • Detects which application/window is active
  • Does NOT capture screenshots
  • Stores identifiers and hashes only

♫ AUDIO MONITORING
  • Detects audio activity
  • May transcribe speech if enabled
  • Audio is processed locally
  • Raw audio is not stored

⚙ SYSTEM ACCESS
  • Allows basic system awareness (files, processes)
  • Required for deeper integration
  • May trigger OS permission prompts

Permissions can be revoked at any time.

================================================================================
                      JOURNALS & TRANSPARENCY
================================================================================

The system keeps logs for accountability and review:

  • Conversation log — what was said
  • System log — what the system did and why

If logs are modified after the fact, the system may pause
and require re-consent before continuing.

================================================================================
                        HOW LEARNING WORKS
================================================================================

Learning happens in stages:

  1. MIMIC — observing new patterns
  2. DAYDREAM — forming relationships
  3. RESPOND — acting with higher confidence

Confidence builds slowly and can decrease.

================================================================================
                   WHAT THIS SYSTEM WILL NOT DO
================================================================================

By design, it will not:
  ✗ Access the internet without instruction
  ✗ Run invisibly or in hidden mode
  ✗ Hide its decision-making process
  ✗ Act without permission

================================================================================
                           DATA HANDLING
================================================================================

All data remains local:
  • No cloud storage
  • No telemetry
  • No third-party access

You are responsible for securing these files.

================================================================================
                           ACCEPTANCE
================================================================================

By accepting, you confirm that:
  • You understand what the system can and cannot do
  • You accept responsibility for its use
  • You agree to transparent logging

If you are not comfortable with this, decline and exit.
"""

class TermsDialog(QWidget):
    accepted = Signal()
    declined = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Bootstrap - Terms of Use")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(*TERMS_WINDOW_SIZE)

        self.permissions = {
            "keyboard_monitoring": False,
            "screen_monitoring": False,
            "audio_monitoring": False,
            "system_admin": False
        }

        self._dragPos = None
        self._initUI()

    def _initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("⍟ System Bootstrap Agreement")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLOR_SYS};")
        layout.addWidget(title)

        textArea = QTextEdit()
        textArea.setReadOnly(True)
        textArea.setFont(QFont("Consolas", 10))
        textArea.setPlainText(AGREEMENT_TEXT)
        textArea.setStyleSheet(f"background: rgba(26,26,26,0.6); color:{COLOR_TEXT}; border-radius:8px; padding:12px;")
        layout.addWidget(textArea)

        permLayout = QVBoxLayout()
        permLabel = QLabel("Grant Permissions:")
        permLabel.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        permLabel.setStyleSheet(f"color: {COLOR_WARNING};")
        permLayout.addWidget(permLabel)

        self.cbKeyboard = QCheckBox("⌨ Enable Keyboard Monitoring" + ("" if PYNPUT_OK else " (requires pynput)"))
        self.cbKeyboard.setStyleSheet(f"color: {COLOR_TEXT};")
        self.cbKeyboard.setEnabled(PYNPUT_OK)
        self.cbKeyboard.toggled.connect(lambda checked: self._setPermission("keyboard_monitoring", checked))
        permLayout.addWidget(self.cbKeyboard)

        self.cbScreen = QCheckBox("▣ Enable Screen / Activity Monitoring")
        self.cbScreen.setStyleSheet(f"color: {COLOR_TEXT};")
        self.cbScreen.toggled.connect(lambda checked: self._setPermission("screen_monitoring", checked))
        permLayout.addWidget(self.cbScreen)

        self.cbAudio = QCheckBox("♫ Enable Audio Monitoring" + ("" if PYAUDIO_OK else " (requires pyaudio)"))
        self.cbAudio.setStyleSheet(f"color: {COLOR_TEXT};")
        self.cbAudio.setEnabled(PYAUDIO_OK)
        self.cbAudio.toggled.connect(lambda checked: self._setPermission("audio_monitoring", checked))
        permLayout.addWidget(self.cbAudio)

        self.cbAdmin = QCheckBox("⚙ Allow System Administration")
        self.cbAdmin.setStyleSheet(f"color: {COLOR_TEXT};")
        self.cbAdmin.toggled.connect(lambda checked: self._setPermission("system_admin", checked))
        permLayout.addWidget(self.cbAdmin)

        layout.addLayout(permLayout)

        btnLayout = QHBoxLayout()
        declineBtn = QPushButton("DECLINE")
        declineBtn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        declineBtn.clicked.connect(self._onDecline)
        declineBtn.setStyleSheet(f"background:{COLOR_DECLINE}; color:#fff; border:0; border-radius:7px; padding:10px 26px;")
        btnLayout.addWidget(declineBtn)

        btnLayout.addStretch()

        acceptBtn = QPushButton("ACCEPT SHEPHERD ROLE")
        acceptBtn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        acceptBtn.clicked.connect(self._onAccept)
        acceptBtn.setStyleSheet(f"background:{COLOR_ACCEPT}; color:#000; border:0; border-radius:7px; padding:10px 26px;")
        btnLayout.addWidget(acceptBtn)

        layout.addLayout(btnLayout)

        self.setStyleSheet("""TermsDialog {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(17, 24, 39, 0.92),
                stop:1 rgba(31, 41, 55, 0.92)
            );
            border-radius: 15px;
        }""")

    def _setPermission(self, key: str, value: bool):
        self.permissions[key] = value

    def _onAccept(self):
        self.accepted.emit()

    def _onDecline(self):
        self.declined.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragPos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._dragPos:
            self.move(event.globalPosition().toPoint() - self._dragPos)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragPos = None


# ============================================================
# Main System Window
# ============================================================

class SystemWindow(QWidget):
    def __init__(self, log: EventLog, config: ConfigManager, learning_path: Path):
        super().__init__()
        self.log = log
        self.config = config
        self.learning_path = learning_path

        self.learning = ThreeStageLearning()
        self.learning.load(self.learning_path)

        self.keyboardMonitor = KeyboardMonitor()
        self.screenMonitor = ScreenMonitor()
        self.audioMonitor = AudioMonitor()

        self.keyboardMonitor.keyPressed.connect(self._onKeyPress)
        self.keyboardMonitor.typingPattern.connect(self._onTypingPattern)
        self.screenMonitor.windowActivity.connect(self._onWindowActivity)
        self.audioMonitor.audioDetected.connect(self._onAudioDetected)
        self.audioMonitor.speechDetected.connect(self._onSpeechDetected)

        self.isBootstrapping = not self.config.isInitialized()
        self.bootstrapStep = 0
        self.shepherdName = self.config.data.get("shepherd_name")
        self.systemName = self.config.data.get("system_name")

        self.sensoryContext = {
            "keyboard": {"active": False, "lastPattern": None, "lastTs": 0.0},
            "screen": {"changed": False, "process": None, "title": None, "lastTs": 0.0},
            "audio": {"active": False, "level": 0.0, "lastTs": 0.0}
        }

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.resize(*SYSTEM_WINDOW_SIZE)

        self._dragPos = None
        self._resizing = False
        self._resizeStartPos = None
        self._resizeStartGeo = None

        self._initUI()
        self._startSensors()

        self.saveTimer = QTimer()
        self.saveTimer.timeout.connect(self._autoSave)
        self.saveTimer.start(30000)

        self.decayTimer = QTimer()
        self.decayTimer.timeout.connect(self._applyDecay)
        self.decayTimer.start(500)

    def _initUI(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)

        topRow = QHBoxLayout()
        titleText = "⍟ System Bootstrap" if self.isBootstrapping else f"⍟ {self.systemName}"
        self.title = QLabel(titleText)
        self.title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.title.setStyleSheet(f"color: {COLOR_SYS};")
        topRow.addWidget(self.title)
        topRow.addStretch()

        self.btnStopSensors = QPushButton("STOP SENSORS")
        self.btnStopSensors.setStyleSheet("padding:8px 12px; border-radius:10px;")
        self.btnStopSensors.clicked.connect(self._stopSensors)
        topRow.addWidget(self.btnStopSensors)

        self.btnRevoke = QPushButton("REVOKE CONSENT")
        self.btnRevoke.setStyleSheet(f"padding:8px 12px; border-radius:10px; background:{COLOR_DECLINE}; color:#fff;")
        self.btnRevoke.clicked.connect(self._revokeConsent)
        topRow.addWidget(self.btnRevoke)

        layout.addLayout(topRow)

        self.input = QLineEdit()
        self.input.setFont(QFont("Arial", 11))
        self.input.returnPressed.connect(self._onSend)
        self.input.setPlaceholderText("type your name and press Enter..." if self.isBootstrapping else "type something...")
        self.input.setStyleSheet(f"""QLineEdit {{
            background: {BG_INPUT};
            color: {COLOR_TEXT};
            border: 0px;
            border-radius: 10px;
            padding: 12px 15px;
            font-size: 16px;
        }} QLineEdit::placeholder {{ color: rgba(230, 230, 230, 0.35); }}""")
        layout.addWidget(self.input)

        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setFont(QFont("Consolas", 11))
        self.display.setFrameStyle(QTextEdit.Shape.NoFrame)
        self.display.setStyleSheet(f"""QTextEdit {{
            background: rgba(11, 15, 20, 0.45);
            color: {COLOR_TEXT};
            border: 0px;
            border-radius: 10px;
            padding: 8px;
        }}""")
        layout.addWidget(self.display, 1)

        self.debugLabel = QLabel("")
        self.debugLabel.setWordWrap(True)
        self.debugLabel.setFont(QFont("Consolas", 9))
        self.debugLabel.setStyleSheet(f"""QLabel {{
            background: {BG_PANEL};
            color: rgba(230,230,230,0.85);
            padding: 10px 12px;
            border-radius: 10px;
        }}""")
        layout.addWidget(self.debugLabel)

        self.setStyleSheet("""SystemWindow {
            background: qlineargradient(
                x1:0, y1:0, x2:1, y2:1,
                stop:0 rgba(17, 24, 39, 0.85),
                stop:1 rgba(31, 41, 55, 0.85)
            );
            border-radius: 15px;
        }""")

        if self.isBootstrapping:
            self.append("sys", "System requires initialization.")
            self.append("sys", "What is your name? (Shepherd)")
        else:
            self.append("sys", f"Welcome back, {self.shepherdName}.")
            self.append("sys", f"System: {self.systemName}")
            self._showSensoryStatus()

    def _startSensors(self):
        perms = self.config.data.get("permissions", {})
        if perms.get("keyboard_monitoring") and self.keyboardMonitor.start():
            self.log.sensor("keyboard", {"status": "active"})
        if perms.get("screen_monitoring") and self.screenMonitor.start():
            self.log.sensor("screen", {"status": "active"})
        if perms.get("audio_monitoring") and self.audioMonitor.start():
            self.log.sensor("audio", {"status": "active"})

    def _stopSensors(self):
        self.keyboardMonitor.stop()
        self.screenMonitor.stop()
        self.audioMonitor.stop()
        self.log.config_event("sensors_stopped", {"by": "user"})
        self.append("sys", "All sensors stopped.")
        self._showSensoryStatus()

    def _revokeConsent(self):
        self._stopSensors()
        self.config.revoke_all_permissions()
        createConsentRecord(False)
        self.log.consent(False, self.config.data.get("permissions", {}))
        self.append("sys", "Consent revoked. Permissions set to FALSE. Restart required to re-consent.")

    def _showSensoryStatus(self):
        perms = self.config.data.get("permissions", {})
        active = []
        if perms.get("keyboard_monitoring") and self.keyboardMonitor.enabled: active.append("⌨")
        if perms.get("screen_monitoring") and self.screenMonitor.enabled: active.append("▣")
        if perms.get("audio_monitoring") and self.audioMonitor.enabled: active.append("♫")
        self.append("sys", f"Active senses: {' '.join(active) if active else '(none)'}")

    def _onSend(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()
        self.append("user", text)
        self.log.conversation(self.shepherdName or "USER", text)

        if self.isBootstrapping:
            self._handleBootstrap(text)
            return

        stage, response, depictive, debug = self.learning.process(text, self._context_snapshot())
        color = {"mimic": COLOR_MIMIC, "daydream": COLOR_DAYDREAM, "respond": COLOR_RESPOND}.get(stage, COLOR_SYS)
        self.appendColored(f"sys:{stage}", response, color, depictive)
        self._setDebug(debug)
        self.log.system(stage, response, debug, depictive)

    def _handleBootstrap(self, text: str):
        if self.bootstrapStep == 0:
            self.shepherdName = text
            self.append("sys", f"Welcome, {self.shepherdName}.")
            self.append("sys", "What will you name this system?")
            self.input.setPlaceholderText("type system name and press Enter...")
            self.bootstrapStep = 1
            self.log.config_event("bootstrap_shepherd", {"shepherd": self.shepherdName})
        elif self.bootstrapStep == 1:
            self.systemName = text
            permissions = self.config.data.get("permissions", {})
            self.config.initialize(self.shepherdName, self.systemName, permissions)
            self.isBootstrapping = False
            self.title.setText(f"⍟ {self.systemName}")
            self.append("sys", f"{self.systemName} initialized.")
            self.append("sys", "Bootstrap complete. Shepherd Protocol active.")
            self.input.setPlaceholderText("type something...")
            self.log.config_event("bootstrap_complete", {"shepherd": self.shepherdName, "system": self.systemName, "permissions": permissions})
            self._showSensoryStatus()

    def _setDebug(self, debug: Dict[str, Any]):
        s = debug.get("sensors", {})
        self.debugLabel.setText(
            f"mode={debug.get('stage')} | seen={debug.get('seen')} | conf={debug.get('confidence')} | "
            f"rels={debug.get('relations')} | best={debug.get('best_match')} | kbd={s.get('keyboard_active')} scr={s.get('screen_process')} aud={s.get('audio_active')}"
        )

    def _context_snapshot(self) -> Dict[str, Any]:
        return {
            "keyboard": {"active": bool(self.sensoryContext["keyboard"]["active"]), "lastPattern": self.sensoryContext["keyboard"]["lastPattern"]},
            "screen": {"changed": bool(self.sensoryContext["screen"]["changed"]), "process": self.sensoryContext["screen"]["process"], "title": self.sensoryContext["screen"]["title"]},
            "audio": {"active": bool(self.sensoryContext["audio"]["active"]), "level": float(self.sensoryContext["audio"]["level"])}
        }

    def append(self, role: str, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        c = {"user": COLOR_USER, "sys": COLOR_SYS}.get(role, COLOR_TEXT)
        self.display.append(f'<span style="color: {COLOR_TIMESTAMP};">[{ts}] </span><span style="color:{c};">{htmlEscape(role)} &gt; {htmlEscape(text)}</span>')
        self._scrollToBottom()

    def appendColored(self, role: str, text: str, color: str, prefix: str = ""):
        ts = datetime.now().strftime("%H:%M:%S")
        self.display.append(f'<span style="color: {COLOR_TIMESTAMP};">[{ts}] </span><span style="color:{color};">{htmlEscape(prefix)} {htmlEscape(role)} &gt; {htmlEscape(text)}</span>')
        self._scrollToBottom()

    def _scrollToBottom(self):
        self.display.moveCursor(QTextCursor.MoveOperation.End)
        self.display.ensureCursorVisible()

    def _onKeyPress(self, keyClass: str, timestamp: float):
        self.sensoryContext["keyboard"]["active"] = True
        self.sensoryContext["keyboard"]["lastTs"] = timestamp
        self.log.sensor("keyboard", {"event": "keypress", "class": keyClass, "t": timestamp})

    def _onTypingPattern(self, pattern: Dict[str, Any]):
        self.sensoryContext["keyboard"]["lastPattern"] = pattern
        self.log.sensor("keyboard", {"event": "typing_pattern", **pattern})

    def _onWindowActivity(self, data: Dict[str, Any]):
        self.sensoryContext["screen"]["changed"] = bool(data.get("changed"))
        self.sensoryContext["screen"]["process"] = data.get("process")
        self.sensoryContext["screen"]["title"] = data.get("title")
        self.sensoryContext["screen"]["lastTs"] = float(data.get("timestamp", now_ts()))
        self.log.sensor("screen", {"event": "window_activity", "process": data.get("process"), "title": data.get("title"), "changed": data.get("changed"), "t": data.get("timestamp")})

    def _onAudioDetected(self, data: Dict[str, Any]):
        self.sensoryContext["audio"]["active"] = True
        self.sensoryContext["audio"]["level"] = float(data.get("level", 0.0))
        self.sensoryContext["audio"]["lastTs"] = float(data.get("timestamp", now_ts()))
        self.log.sensor("audio", {"event": "audio_activity", **data})

    def _onSpeechDetected(self, text: str):
        self.append("sys", f"♫ Heard: {text}")
        self.log.conversation("AUDIO", text)

    def _applyDecay(self):
        t = now_ts()
        if self.sensoryContext["keyboard"]["active"] and (t - float(self.sensoryContext["keyboard"]["lastTs"])) * 1000.0 > KEYBOARD_ACTIVE_DECAY_MS:
            self.sensoryContext["keyboard"]["active"] = False
        if self.sensoryContext["audio"]["active"] and (t - float(self.sensoryContext["audio"]["lastTs"])) * 1000.0 > AUDIO_ACTIVE_DECAY_MS:
            self.sensoryContext["audio"]["active"] = False

    def _autoSave(self):
        self.learning.save(self.learning_path)
        self.log.render_markdown_view(max_lines=4000)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragPos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        elif event.button() == Qt.MouseButton.RightButton:
            self._resizing = True
            self._resizeStartPos = event.globalPosition().toPoint()
            self._resizeStartGeo = self.geometry()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._dragPos:
            self.move(event.globalPosition().toPoint() - self._dragPos)
        elif (event.buttons() & Qt.MouseButton.RightButton) and self._resizing:
            delta = event.globalPosition().toPoint() - self._resizeStartPos
            newW = max(SYSTEM_MIN_SIZE[0], self._resizeStartGeo.width() + delta.x())
            newH = max(SYSTEM_MIN_SIZE[1], self._resizeStartGeo.height() + delta.y())
            self.resize(newW, newH)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragPos = None
        elif event.button() == Qt.MouseButton.RightButton:
            self._resizing = False

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_Escape:
            self.close()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self._stopSensors()
        self.learning.save(self.learning_path)
        self.log.close("User closed window")
        self.log.render_markdown_view(max_lines=4000)
        super().closeEvent(event)


# ============================================================
# Entry Point
# ============================================================

def new_session_dir() -> Path:
    sid = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    return d

def main():
    app = QApplication(sys.argv)
    termsDialog = TermsDialog()
    config = None
    log = None
    systemWindow = None

    def onAccepted():
        nonlocal config, log, systemWindow
        print("✓ Shepherd role accepted")

        config = ConfigManager(CONFIG_PATH)
        session_dir = new_session_dir()
        log = EventLog(session_dir)

        config.data["permissions"] = termsDialog.permissions
        config.save()

        createConsentRecord(True)
        log.consent(True, termsDialog.permissions)

        learning_path = session_dir / "learning_state.json"
        systemWindow = SystemWindow(log, config, learning_path)

        termsDialog.close()
        systemWindow.show()

    def onDeclined():
        print("✗ Shepherd role declined")
        createConsentRecord(False)
        session_dir = new_session_dir()
        log = EventLog(session_dir)
        log.consent(False, termsDialog.permissions)
        log.close("Declined at terms")
        log.render_markdown_view(max_lines=1000)
        termsDialog.close()
        app.quit()

    termsDialog.accepted.connect(onAccepted)
    termsDialog.declined.connect(onDeclined)
    termsDialog.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
