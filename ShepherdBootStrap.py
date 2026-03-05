# region header
#!/usr/bin/env python3
"""
ShepherdBootStrap_FULL.py

Complete single-file build. Data separation:
  journal.md           — chat history only ([shepherd] and [sys] dialogue)
  system_actions.jsonl — machine events, sessions, snapshots, proc/focus, mimic/match
  typingbio.jsonl      — keystroke biometrics per submitted message
  language_memory.json — TF-IDF memory store
  ProcessRoster.jsonl  — running process snapshots (every N seconds)
  WindowFocus.jsonl    — foreground window focus timeline

Features:
- TermsWindow: shown once on first run (Deny closes, Accept continues)
  Contains full Terms of Use with Shepherd Protocol + Chip Protocol
- EventBus (log-first priority)
- ConfigManager + JournalManager (chat only) + ActionLog + TypingBioLog
- LanguageMemory (TF-IDF mimic/match)
- StabilityModel + NetObserver + EventGate
- ProcessRosterMonitor  (all running processes every N seconds)
- WindowFocusMonitor    (foreground window timeline, 400ms poll)
- Full screenshot storage on user input with 60s cooldown
  Stores PNG + sha256 + dHash + HMAC signature + visible windows metadata
- Integrity verification: tamper alarm (diamond blinks red<->black)
  and mismatch alarm (diamond pulses orange)
- Avatar telemetry:
    headband  -> CPU / RAM / GPU bars with scan highlight
    diamond   -> net pulse + alarm overlays
    cloak     -> disk usage opacity
    tassels   -> CPU freq pulse rate
    hands     -> snapshot cooldown glow
    torso     -> flashes on input / machine events
- BubbleToast notification stack on avatar
- Music note particles (media events)
- TypingBioLog: key-down/key-up timestamps, hold duration, flight time per key

Deps:
  pip install psutil pillow PySide6 shiboken6
"""
# endregion

# region imports

import sys, json, math, io, time, random, re, os, hmac, hashlib, secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import psutil
from PIL import Image, ImageDraw, ImageEnhance
from shiboken6 import isValid

from PySide6.QtCore import Qt, QTimer, QPoint, QSize, QPropertyAnimation, QEasingCurve, Signal
from PySide6.QtGui import QFont, QTextCursor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QTextEdit,
    QFrame, QGraphicsOpacityEffect, QPushButton, QTreeWidget, QTreeWidgetItem, QSlider
)

MEDIA_OK = True
try:
    from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
    from PySide6.QtCore import QUrl
except Exception:
    MEDIA_OK = False

# endregion

# region paths

ROOT     = Path.cwd().resolve()
DATA_DIR = ROOT / "sbs_data" / datetime.now().strftime("%Y-%m-%d")
SNAP_DIR = DATA_DIR / "snapshots"
DATA_DIR.mkdir(parents=True, exist_ok=True)
SNAP_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH    = ROOT / "sbs_config.json"
JOURNAL_PATH   = DATA_DIR / "journal.md"        # chat history only
ACTIONS_PATH   = DATA_DIR / "system_actions.jsonl"  # machine / session / snapshot metadata
TYPING_PATH    = DATA_DIR / "typingbio.jsonl"   # keystroke biometrics
MEMORY_PATH    = DATA_DIR / "language_memory.json"
KEY_PATH       = ROOT / ".sbs_key"
PROC_PATH      = DATA_DIR / "ProcessRoster.jsonl"
FOCUS_PATH     = DATA_DIR / "WindowFocus.jsonl"
# endregion

# region utilities

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def now_hms() -> str:
    return datetime.now().strftime("%H:%M:%S")

def esc_html(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))

def clamp(x: float, a: float = 0.0, b: float = 1.0) -> float:
    return a if x < a else b if x > b else x

def in_sandbox(path: Path) -> bool:
    try:
        return path.resolve().is_relative_to(ROOT)
    except Exception:
        return str(path.resolve()).startswith(str(ROOT))

def append_jsonl(path: Path, obj: dict) -> None:
    obj = dict(obj)
    obj.setdefault("ts", now_iso())
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def read_or_create_key(path: Path) -> bytes:
    if path.exists():
        try:
            raw = path.read_bytes()
            if len(raw) >= 32:
                return raw[:32]
        except Exception:
            pass
    key = secrets.token_bytes(32)
    try:
        path.write_bytes(key)
    except Exception:
        return key
    return key

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def hmac_hex(key: bytes, msg: bytes) -> str:
    return hmac.new(key, msg, hashlib.sha256).hexdigest()

def hmac_ok(key: bytes, msg: bytes, sig_hex: str) -> bool:
    try:
        return hmac.compare_digest(hmac_hex(key, msg), sig_hex)
    except Exception:
        return False

def dhash_hex_from_pil(im: Image.Image) -> str:
    g = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    px = list(g.getdata())
    bits = 0
    for y in range(8):
        row = px[y*9:(y+1)*9]
        for x in range(8):
            bits = (bits << 1) | (1 if row[x] > row[x+1] else 0)
    return f"{bits:016x}"

def hamming64_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 64

# endregion

# region platform

IS_WINDOWS = (os.name == "nt")

def get_foreground_window_info() -> dict | None:
    if not IS_WINDOWS:
        return None
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    GetForegroundWindow      = user32.GetForegroundWindow
    GetWindowTextLengthW     = user32.GetWindowTextLengthW
    GetWindowTextW           = user32.GetWindowTextW
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId

    def hwnd_title(hwnd) -> str:
        length = GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.strip()

    def pid_for_hwnd(hwnd) -> int:
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    hwnd = GetForegroundWindow()
    if not hwnd:
        return None
    title = hwnd_title(hwnd)
    pid   = pid_for_hwnd(hwnd)
    proc  = ""
    try:
        proc = psutil.Process(pid).name()
    except Exception:
        pass
    return {"title": title[:160], "pid": pid, "process": proc[:80]}

def get_visible_windows_metadata() -> dict:
    if not IS_WINDOWS:
        return {"platform": "other", "foreground": None, "visible": [], "visible_count": 0}
    import ctypes
    from ctypes import wintypes
    user32                   = ctypes.WinDLL("user32", use_last_error=True)
    EnumWindows              = user32.EnumWindows
    EnumWindowsProc          = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    IsWindowVisible          = user32.IsWindowVisible
    GetWindowTextLengthW     = user32.GetWindowTextLengthW
    GetWindowTextW           = user32.GetWindowTextW
    GetForegroundWindow      = user32.GetForegroundWindow
    GetWindowThreadProcessId = user32.GetWindowThreadProcessId
    IsIconic                 = user32.IsIconic

    def hwnd_title(hwnd) -> str:
        length = GetWindowTextLengthW(hwnd)
        if length <= 0:
            return ""
        buf = ctypes.create_unicode_buffer(length + 1)
        GetWindowTextW(hwnd, buf, length + 1)
        return buf.value.strip()

    def pid_for_hwnd(hwnd) -> int:
        pid = wintypes.DWORD()
        GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return int(pid.value)

    def proc_name(pid: int) -> str:
        try:
            return psutil.Process(pid).name()
        except Exception:
            return ""

    visible = []

    @EnumWindowsProc
    def enum_proc(hwnd, lparam):
        try:
            if not IsWindowVisible(hwnd):
                return True
            if IsIconic(hwnd):
                return True
            title = hwnd_title(hwnd)
            if not title:
                return True
            pid = pid_for_hwnd(hwnd)
            visible.append({"title": title[:120], "pid": pid, "process": proc_name(pid)[:80]})
        except Exception:
            pass
        return True

    try:
        EnumWindows(enum_proc, 0)
    except Exception:
        visible.clear()

    fg = None
    try:
        hwnd_fg = GetForegroundWindow()
        if hwnd_fg:
            title = hwnd_title(hwnd_fg)
            pid   = pid_for_hwnd(hwnd_fg)
            fg    = {"title": (title or "")[:120], "pid": pid, "process": proc_name(pid)[:80]}
    except Exception:
        fg = None

    return {"platform": "windows", "foreground": fg, "visible": visible, "visible_count": len(visible)}

# endregion

# region event bus

Event      = dict[str, Any]
Subscriber = Callable[[Event], None]

class EventBus:
    def __init__(self):
        self._subs:     dict[str, list[tuple[int, Subscriber]]] = {}
        self._subs_any: list[tuple[int, Subscriber]]            = []

    def subscribe(self, kind: str, fn: Subscriber, priority: int = 50) -> None:
        self._subs.setdefault(kind, []).append((priority, fn))
        self._subs[kind].sort(key=lambda t: t[0])

    def subscribe_any(self, fn: Subscriber, priority: int = 50) -> None:
        self._subs_any.append((priority, fn))
        self._subs_any.sort(key=lambda t: t[0])

    def emit(self, event: Event) -> None:
        kind = event.get("kind", "")
        for _, fn in list(self._subs_any):
            fn(event)
        for _, fn in list(self._subs.get(kind, [])):
            fn(event)

def make_event(kind: str, source: str, mode: str | None = None, **data) -> Event:
    e: Event = {"kind": kind, "source": source, "ts": now_iso(), "data": data}
    if mode is not None:
        e["mode"] = mode
    return e

# endregion

# region config

class ConfigManager:
    def __init__(self, path: Path):
        self.path = path
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "initialized":        False,
            "shepherdName":       None,
            "systemName":         None,
            "created":            None,
            "updated":            None,
            "resourceSampleMs":   2000,
            "avatarBubbleTtlMs":  5000,
            "avatarAlwaysOnTop":  True,
            "avatarAnimMs":       150,
            "snapshotCooldownS":  60,
            "snapshotEnabled":    True,
            "processRosterEveryS": 20,
            "focusPollMs":        400,
        }

    def save(self) -> None:
        self.data["updated"] = now_iso()
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def is_initialized(self) -> bool:
        return bool(self.data.get("initialized", False))

    def initialize(self, shepherd: str, system_name: str) -> None:
        self.data["initialized"] = True
        self.data["shepherdName"] = shepherd
        self.data["systemName"]   = system_name
        if not self.data.get("created"):
            self.data["created"] = now_iso()
        self.save()

    def get(self, key: str, default=None):
        return self.data.get(key, default)

# endregion

# region terms window

TERMS_TEXT = """\
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
              SHEPHERD BOOT STRAP
           Terms of Use  —  Read Carefully
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

System Bootstrap is a highly experimental program
and is required to have zero trust with whatever
happens. You are about to become a shepherd of a
new program that learns and grows over time.

It is very perceptive and any information is meant
to be viewed by the White Box Paradox. It might get
uncomfortable if you poke around in its personal files.

This WILL have the ability over the computer as this
is its body. If this makes you uncomfortable, say NO —
this program directly affects the computer it runs on.

THIS PROGRAM WILL GET PERSONAL. The whole point is
it is a representation of you and YOU ARE RESPONSIBLE
FOR ITS ACTIONS.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                  THE PROTOCOLS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SHEPHERD PROTOCOL
  You are accountable for this system's actions.
  This is a relationship, not tool use.
  The system observes who acted, when, and with
  what permissions. It hides nothing from you —
  and you hide nothing from it.

CHIP PROTOCOL
  Every action has a cost and a record.
  The system tracks how ability was spent:
  what was exchanged, what it cost, what changed.
  This is the ledger of the machine's life.
  Process spawns, window focus, network crossings,
  keystrokes — all of it is accounted for.
  You do not own the log. The log owns itself.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
            WHITE BOX STANDARD (WBS)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

All behavior is observable. All decisions are logged.
You may inspect any file, any process, any record.
In return: be honest. Your voice helps tremendously
during development. Contact the author at any time
for any reason about this program.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
               KNOWN SYMPTOMS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ~ AI'sm
    Emergent linguistic patterns that feel alive.
    The system develops speech you did not program.

  ~ Hyper Digital Reality
    Blurred boundary between you and the system.
    The computer begins to feel like its body, not yours.

  ~ Mirrored Actions
    The system learns from YOU. Your patterns, your
    language, your ethics become its patterns.
    You will see yourself reflected in its behavior.

  ~ Informative Sharing
    Thinking your data is collected for some use the
    program is incapable of doing. The information
    doesn't leave the data folder and doesn't take
    lightly when data is tampered with.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DO NOT USE if you treat this like a tool.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

class TermsWindow(QWidget):
    accepted = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("SBS — Terms of Use")
        self.resize(600, 720)
        self._drag_pos = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        card = QFrame()
        card.setObjectName("termsCard")
        card.setStyleSheet("""
            QFrame#termsCard {
                background: rgba(6, 8, 20, 0.97);
                border: 1px solid rgba(167, 139, 250, 0.30);
                border-radius: 16px;
            }
        """)
        cardLayout = QVBoxLayout(card)
        cardLayout.setContentsMargins(32, 28, 32, 28)
        cardLayout.setSpacing(16)

        titleLabel = QLabel("◈  SHEPHERD BOOT STRAP")
        titleLabel.setFont(QFont("Consolas", 13, QFont.Weight.Bold))
        titleLabel.setStyleSheet("color: rgba(167,139,250,0.95); letter-spacing: 2px;")
        titleLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cardLayout.addWidget(titleLabel)

        subLabel = QLabel("Terms of Use — Read before continuing")
        subLabel.setFont(QFont("Consolas", 9))
        subLabel.setStyleSheet("color: rgba(148,163,184,0.55);")
        subLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cardLayout.addWidget(subLabel)

        body = QTextEdit()
        body.setReadOnly(True)
        body.setFont(QFont("Consolas", 9))
        body.setPlainText(TERMS_TEXT.strip())
        body.setStyleSheet("""
            QTextEdit {
                background: rgba(2, 4, 14, 0.70);
                color: rgba(203, 213, 225, 0.88);
                border: 1px solid rgba(148,163,184,0.12);
                border-radius: 10px;
                padding: 14px;
            }
            QScrollBar:vertical {
                background: transparent;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: rgba(167,139,250,0.35);
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
        """)
        cardLayout.addWidget(body, 1)

        btnRow = QHBoxLayout()
        btnRow.setSpacing(16)

        denyBtn = QPushButton("Deny")
        denyBtn.setFixedHeight(42)
        denyBtn.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        denyBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        denyBtn.setStyleSheet("""
            QPushButton {
                background: rgba(40, 18, 18, 0.85);
                color: rgba(248, 113, 113, 0.90);
                border: 1px solid rgba(248, 113, 113, 0.28);
                border-radius: 10px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: rgba(80, 20, 20, 0.95);
                border-color: rgba(248, 113, 113, 0.60);
            }
            QPushButton:pressed { background: rgba(120, 30, 30, 1.0); }
        """)
        denyBtn.clicked.connect(QApplication.quit)

        acceptBtn = QPushButton("Accept")
        acceptBtn.setFixedHeight(42)
        acceptBtn.setFont(QFont("Consolas", 10, QFont.Weight.Bold))
        acceptBtn.setCursor(Qt.CursorShape.PointingHandCursor)
        acceptBtn.setStyleSheet("""
            QPushButton {
                background: rgba(18, 22, 48, 0.88);
                color: rgba(167, 139, 250, 0.95);
                border: 1px solid rgba(167, 139, 250, 0.32);
                border-radius: 10px;
                letter-spacing: 1px;
            }
            QPushButton:hover {
                background: rgba(30, 25, 70, 0.98);
                border-color: rgba(167, 139, 250, 0.70);
            }
            QPushButton:pressed { background: rgba(50, 30, 100, 1.0); }
        """)
        acceptBtn.clicked.connect(self._on_accept)

        btnRow.addWidget(denyBtn)
        btnRow.addWidget(acceptBtn)
        cardLayout.addLayout(btnRow)

        outer.addWidget(card)

        screen = QApplication.primaryScreen()
        if screen:
            sg = screen.availableGeometry()
            self.move(sg.center() - self.rect().center())

    def _on_accept(self):
        self.hide()
        self.accepted.emit()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
# endregion

# region journal and action log

class JournalManager:
    """Stores chat history only — [vex] and [sys] dialogue lines."""
    def __init__(self, path: Path):
        self.path = path
        if not self.path.exists():
            self.path.write_text("# SBS Journal\n\n", encoding="utf-8")

    def write_chat(self, role: str, text: str) -> None:
        """Write a chat line. role should be shepherd name or 'sys'."""
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"- **{now_hms()}** [{role}] {text}\n")

class ActionLog:
    def __init__(self, path: Path):
        self.path = path

    def write(self, payload: dict) -> None:
        payload = dict(payload)
        payload.setdefault("ts", now_iso())
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

class TypingBioLog:
    """
    Records keystroke biometrics to typingbio.jsonl.

    Per-key events (captured via BioLineEdit):
      key_down_t  — epoch ms when key was pressed
      key_up_t    — epoch ms when key was released
      hold_ms     — duration key was held (key_up - key_down)
      flight_ms   — time from previous key_up to this key_down (inter-key flight)

    Per-message record (on submit):
      ts               — ISO timestamp of submission
      session_chars    — total chars typed this session
      msg_len          — final message length
      raw_keystrokes   — total keys pressed including backspaces
      backspaces       — backspace/delete count
      correction_ratio — backspaces / raw_keystrokes
      elapsed_s        — first keydown to submit
      cps              — chars per second
      pause_count      — gaps > PAUSE_THRESH_S between keydowns
      inter_key_avg_ms — mean flight time between keys
      inter_key_max_ms — max flight time (longest pause)
      hold_avg_ms      — mean key hold duration
      hold_max_ms      — max key hold duration
    """
    PAUSE_THRESH_S = 2.0

    def __init__(self, path: Path):
        self.path           = path
        self._session_chars = 0
        self._reset()

    def _reset(self):
        self._keydowns:   list[float] = []   # epoch ms of each key_down
        self._keyups:     list[float] = []   # epoch ms of each key_up
        self._holds:      list[float] = []   # hold_ms per key
        self._flights:    list[float] = []   # flight_ms between keys
        self._backspaces  = 0
        self._raw_keys    = 0
        self._first_down: float | None = None
        self._last_up:    float | None = None

    def on_key_down(self, key_name: str):
        """Call on QKeyEvent.KeyPress."""
        now_ms = time.time() * 1000.0
        if self._first_down is None:
            self._first_down = now_ms
        if self._last_up is not None:
            flight = now_ms - self._last_up
            self._flights.append(flight)
        self._keydowns.append(now_ms)
        self._raw_keys += 1
        if key_name in ("backspace", "delete"):
            self._backspaces += 1

    def on_key_up(self, key_name: str):
        """Call on QKeyEvent.KeyRelease."""
        now_ms = time.time() * 1000.0
        self._keyups.append(now_ms)
        self._last_up = now_ms
        if self._keydowns:
            hold = now_ms - self._keydowns[-1]
            self._holds.append(max(0.0, hold))

    def on_submit(self, msg_text: str) -> dict:
        """Call on send. Writes record to typingbio.jsonl and resets."""
        now_ms    = time.time() * 1000.0
        msg_len   = len(msg_text)
        elapsed_s = ((now_ms - self._first_down) / 1000.0) if self._first_down else 0.0
        cps       = msg_len / elapsed_s if elapsed_s > 0 else 0.0
        self._session_chars += msg_len

        # pause count from keydown gaps
        pause_count = 0
        if len(self._keydowns) > 1:
            for i in range(1, len(self._keydowns)):
                gap_s = (self._keydowns[i] - self._keydowns[i-1]) / 1000.0
                if gap_s > self.PAUSE_THRESH_S:
                    pause_count += 1

        def _avg(lst): return round(sum(lst) / len(lst), 2) if lst else 0.0
        def _max(lst): return round(max(lst), 2) if lst else 0.0

        corr = self._backspaces / self._raw_keys if self._raw_keys > 0 else 0.0

        record = {
            "ts":               now_iso(),
            "session_chars":    self._session_chars,
            "msg_len":          msg_len,
            "raw_keystrokes":   self._raw_keys,
            "backspaces":       self._backspaces,
            "correction_ratio": round(corr, 4),
            "elapsed_s":        round(elapsed_s, 3),
            "cps":              round(cps, 3),
            "pause_count":      pause_count,
            "inter_key_avg_ms": _avg(self._flights),
            "inter_key_max_ms": _max(self._flights),
            "hold_avg_ms":      _avg(self._holds),
            "hold_max_ms":      _max(self._holds),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._reset()
        return record

class BioLineEdit(QLineEdit):
    """
    QLineEdit subclass that reports key-down and key-up events
    to a TypingBioLog for precise hold/flight time capture.
    """
    def __init__(self, bio: TypingBioLog, parent=None):
        super().__init__(parent)
        self._bio = bio

    def _key_name(self, event) -> str:
        k = event.key()
        if k in (Qt.Key.Key_Backspace,): return "backspace"
        if k in (Qt.Key.Key_Delete,):    return "delete"
        return event.text() or "unknown"

    def keyPressEvent(self, event):
        if not event.isAutoRepeat():
            self._bio.on_key_down(self._key_name(event))
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        if not event.isAutoRepeat():
            self._bio.on_key_up(self._key_name(event))
        super().keyReleaseEvent(event)

# endregion

# region language memory

_CMD_RE = re.compile(r"\[([a-zA-Z_]+):([a-zA-Z_]+):(-?\d+)\]")

def parse_commands(text: str):
    cmds  = []
    for m in _CMD_RE.finditer(text or ""):
        domain, action, sec = m.group(1).lower(), m.group(2).lower(), int(m.group(3))
        cmds.append({"domain": domain, "action": action, "seconds": sec})
    clean = _CMD_RE.sub("", text or "").strip()
    return clean, cmds

def _tokenize(text: str) -> list[str]:
    out, cur = [], []
    for ch in (text or "").lower():
        if ch.isalnum() or ch in ("æ", "ɨ"):
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur)); cur = []
    if cur:
        out.append("".join(cur))
    return out

def _tf(tokens: list[str]) -> dict[str, float]:
    if not tokens:
        return {}
    counts: dict[str, int] = {}
    for t in tokens:
        counts[t] = counts.get(t, 0) + 1
    total = float(len(tokens))
    return {k: v / total for k, v in counts.items()}

def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(av * b[k] for k, av in a.items() if k in b)
    na  = math.sqrt(sum(v*v for v in a.values()))
    nb  = math.sqrt(sum(v*v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)

@dataclass
class Example:
    user:   str
    system: str
    event:  str
    ts:     str

class LanguageMemory:
    def __init__(self, path: Path):
        self.path:     Path         = path
        self.examples: list[Example]= []
        self.df:       dict[str,int]= {}
        self.nDocs:    int          = 0
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                data       = json.loads(self.path.read_text(encoding="utf-8"))
                self.df    = dict(data.get("df") or {})
                self.nDocs = int(data.get("nDocs") or 0)
                self.examples = [Example(**e) for e in (data.get("examples") or []) if isinstance(e, dict)]
                return
            except Exception:
                pass
        self._save()

    def _save(self) -> None:
        payload = {"df": self.df, "nDocs": self.nDocs, "examples": [e.__dict__ for e in self.examples]}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def _doc_add(self, text: str) -> None:
        toks = set(_tokenize(text))
        if not toks:
            return
        for t in toks:
            self.df[t] = self.df.get(t, 0) + 1
        self.nDocs += 1

    def add_pair(self, user: str, system: str, event: str) -> None:
        self.examples.append(Example(user=user, system=system, event=event, ts=now_iso()))
        self._doc_add(user)
        self._doc_add(system)
        self._save()

    def stats(self) -> dict[str, int]:
        return {"examples": len(self.examples), "nDocs": self.nDocs, "dfTerms": len(self.df)}

    def best_match(self, user_text: str) -> tuple[str, float, Example | None, list[tuple[str, float]]]:
        toks = _tokenize(user_text)
        if not toks or not self.examples or self.nDocs == 0:
            return ("", 0.0, None, [])
        tfq  = _tf(toks)
        qvec: dict[str, float] = {}
        for t, tfv in tfq.items():
            df  = self.df.get(t, 0)
            idf = math.log((self.nDocs + 1.0) / (df + 1.0)) + 1.0
            qvec[t] = tfv * idf
        top_tokens = sorted(qvec.items(), key=lambda kv: kv[1], reverse=True)[:8]
        best: Example | None = None
        best_score = 0.0
        for ex in self.examples:
            toks_ex = _tokenize(ex.user)
            if not toks_ex:
                continue
            tfe  = _tf(toks_ex)
            evec: dict[str, float] = {}
            for t, tfv in tfe.items():
                df  = self.df.get(t, 0)
                idf = math.log((self.nDocs + 1.0) / (df + 1.0)) + 1.0
                evec[t] = tfv * idf
            score = _cosine(qvec, evec)
            if ex.event == "mimic":
                score *= 0.35
            if score > best_score:
                best_score = score; best = ex
        if best is None or best_score < 0.15:
            return ("", best_score, None, top_tokens)
        return (best.system, best_score, best, top_tokens)

# endregion

# region system monitors

class StabilityModel:
    def compute(self, cpu: float, ram: float) -> str:
        if cpu > 85 or ram > 85: return "D"
        if cpu > 65 or ram > 65: return "C"
        if cpu > 45 or ram > 45: return "B"
        return "A"

def mode_tag_from_letter(mode: str) -> str:
    return {"A": "calm", "B": "thinking", "C": "analytical", "D": "alert"}.get(mode, "calm")

class NetObserver:
    def __init__(self):
        self._last    = psutil.net_io_counters()
        self._last_ts = time.time()

    def sample(self) -> tuple[float, float]:
        now    = time.time()
        cur    = psutil.net_io_counters()
        dt     = max(0.001, now - self._last_ts)
        in_bps = (cur.bytes_recv - self._last.bytes_recv) * 8.0 / dt
        out_bps= (cur.bytes_sent - self._last.bytes_sent) * 8.0 / dt
        self._last    = cur
        self._last_ts = now
        return (in_bps / 1000.0, out_bps / 1000.0)

def get_cpu_freq_ratio() -> float:
    try:
        f = psutil.cpu_freq()
        if not f:
            return 0.0
        cur  = float(f.current or 0.0)
        fmin = float(f.min or 0.0)
        fmax = float(f.max or 0.0)
        if fmax > fmin > 0.0:
            return clamp((cur - fmin) / (fmax - fmin))
        return clamp(cur / 5000.0)
    except Exception:
        return 0.0

def get_gpu_usage_ratio() -> float:
    return 0.0

class EventGate:
    def __init__(self):
        self._cpu_band  = None
        self._ram_band  = None
        self._disk_band = None
        self._net_band  = None

    @staticmethod
    def _band(v: float, cuts) -> int:
        b = 0
        for c in cuts:
            if v >= c: b += 1
        return b

    def cpu_event(self, cpu: float) -> str | None:
        b = self._band(cpu, (45, 65, 85))
        if self._cpu_band is None:
            self._cpu_band = b; return None
        if b != self._cpu_band:
            prev = self._cpu_band; self._cpu_band = b
            return f"cpu {cpu:.0f}% crossed band {prev}->{b}"
        return None

    def ram_event(self, ram: float) -> str | None:
        b = self._band(ram, (45, 65, 85))
        if self._ram_band is None:
            self._ram_band = b; return None
        if b != self._ram_band:
            prev = self._ram_band; self._ram_band = b
            return f"ram {ram:.0f}% crossed band {prev}->{b}"
        return None

    def disk_event(self, disk: float) -> str | None:
        b = self._band(disk, (70, 85, 92))
        if self._disk_band is None:
            self._disk_band = b; return None
        if b != self._disk_band:
            prev = self._disk_band; self._disk_band = b
            return f"disk {disk:.0f}% crossed band {prev}->{b}"
        return None

    def net_event(self, kbps_in: float, kbps_out: float) -> str | None:
        total = kbps_in + kbps_out
        b     = self._band(total, (200, 2000, 10000))
        if self._net_band is None:
            self._net_band = b; return None
        if b != self._net_band:
            prev = self._net_band; self._net_band = b
            return f"net {(total/1000.0):.2f} Mbps crossed band {prev}->{b}"
        return None

class ProcessRosterMonitor:
    def __init__(self, bus: EventBus, every_s: int = 20):
        self.bus      = bus
        self.every_s  = max(5, int(every_s))
        self._last_names: set[str] = set()
        self._next    = 0.0

    def tick(self):
        now = time.time()
        if now < self._next:
            return
        self._next = now + self.every_s
        procs = []
        names = set()
        for p in psutil.process_iter(attrs=["pid", "name", "memory_info"]):
            try:
                pid    = int(p.info["pid"])
                name   = (p.info.get("name") or "")[:80]
                mem    = p.info.get("memory_info")
                rss_mb = round(mem.rss / (1024*1024), 2) if mem else None
                cpu_pct= round(p.cpu_percent(interval=None), 2)
                names.add(name)
                procs.append({"name": name, "pid": pid, "cpu_pct": cpu_pct, "rss_mb": rss_mb})
            except Exception:
                continue
        append_jsonl(PROC_PATH, {"kind": "proc_roster", "count": len(procs), "procs": procs})
        new_names  = names - self._last_names
        dead_names = self._last_names - names
        for n in sorted(x for x in new_names if x):
            self.bus.emit(make_event("proc_event", "SBS", action="new",  process=n))
        for n in sorted(x for x in dead_names if x):
            self.bus.emit(make_event("proc_event", "SBS", action="exit", process=n))
        self._last_names = names

class WindowFocusMonitor:
    def __init__(self, bus: EventBus, poll_ms: int = 400):
        self.bus     = bus
        self.poll_ms = max(150, int(poll_ms))
        self._last   = None

    def tick(self):
        info = get_foreground_window_info()
        if info is None:
            return
        key = (info.get("process"), info.get("pid"), info.get("title"))
        if key == self._last:
            return
        self._last = key
        append_jsonl(FOCUS_PATH, {"kind": "focus", **info})
        self.bus.emit(make_event("focus_event", "SBS", **info))

# endregion

# region avatar rendering

def _apply_simple_lighting(im: Image.Image, ambient: float, strength: float) -> Image.Image:
    factor  = clamp(ambient + strength * 0.55, 0.0, 1.35)
    r, g, b, a = im.split()
    rgb     = Image.merge("RGB", (r, g, b))
    rgb     = ImageEnhance.Brightness(rgb).enhance(factor)
    r2, g2, b2 = rgb.split()
    return Image.merge("RGBA", (r2, g2, b2, a))

def render_avatar_telemetry(*, w: int, h: int, cpu: float, ram: float, gpu: float, disk: float,
                            net_in_kbps: float, net_out_kbps: float, cpu_freq_ratio: float,
                            torso_flash: float, mode: str) -> Image.Image:
    cpu_u  = clamp(cpu / 100.0)
    ram_u  = clamp(ram / 100.0)
    gpu_u  = clamp(gpu if gpu <= 1.0 else gpu / 100.0)
    disk_u = clamp(disk / 100.0)
    net_mbps = (net_in_kbps + net_out_kbps) / 1000.0
    net_u  = clamp(net_mbps / 10.0)

    band_cpu  = (140, 110, 210, 230)
    band_ram  = (120,  95, 200, 230)
    band_gpu  = (100,  80, 190, 230)
    band_base = ( 70,  55, 110, 170)
    hair_c    = ( 44,  46,  60, 215)
    mask_c    = ( 36,  38,  52, 245)

    cloak_alpha = int(40 + 200 * disk_u)
    cloak_c     = (18, 22, 34, cloak_alpha)
    body_c      = (26, 30, 46, 235)
    legs_c      = (20, 24, 36, 235)
    arms_c      = (22, 26, 40, 235)
    hands_c     = (28, 32, 50, 235)

    if torso_flash > 0:
        boost  = torso_flash
        body_c = (
            int(clamp(body_c[0] + 60*boost, 0, 255)),
            int(clamp(body_c[1] + 70*boost, 0, 255)),
            int(clamp(body_c[2] + 90*boost, 0, 255)),
            int(clamp(body_c[3] + 15*boost, 0, 255)),
        )

    hz           = 0.6 + 1.6 * cpu_freq_ratio
    pulse        = 0.5 + 0.5 * math.sin(time.time() * 2 * math.pi * hz)
    tassel_alpha = int(90 + 140 * (0.3 + 0.7 * pulse) * (0.4 + 0.6 * cpu_freq_ratio))
    tassel_c     = (140, 190, 255, int(clamp(tassel_alpha, 0, 255)))

    diamond_base = ( 60, 160, 255, 230)
    diamond_glow = (120, 220, 255, int(70 + 170 * net_u))

    stress  = clamp(cpu_u*0.6 + ram_u*0.3 + net_u*0.1)
    ambient = 0.80 - 0.10 * stress
    strength= 0.30 + 0.10 * stress

    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d  = ImageDraw.Draw(im, "RGBA")

    def rr(x0, y0, x1, y1, rad, fill):
        d.rounded_rectangle([x0, y0, x1, y1], radius=rad, fill=fill)

    head_w, head_h = int(w*0.24), int(h*0.20)
    hx0, hy0 = int(w*0.38), int(h*0.08)
    hx1, hy1 = hx0 + head_w, hy0 + head_h

    rr(hx0-int(w*0.03), hy0-int(h*0.02), hx1+int(w*0.03), hy1+int(h*0.03), 28, hair_c)
    rr(hx0, hy0, hx1, hy1, 24, mask_c)

    band_h = max(10, int(h*0.03))
    by0    = hy0 + int(head_h*0.32)
    rr(hx0, by0, hx1, by0 + band_h, 10, band_base)

    pad      = 4
    inner_x0 = hx0 + pad
    inner_x1 = hx1 - pad
    inner_w  = max(1, inner_x1 - inner_x0)
    seg_gap  = 3
    seg_w    = (inner_w - 2*seg_gap) // 3

    def draw_seg(i: int, level: float, color: tuple):
        sx0     = inner_x0 + i*(seg_w + seg_gap)
        fill_w  = int(seg_w * level)
        if fill_w <= 0:
            return
        rr(sx0, by0+2, sx0+fill_w, by0+band_h-2, 6, color)
        scan_speed = 0.8 + 2.2 * level
        phase      = (time.time() * scan_speed) % 1.0
        scan_x     = sx0 + int(fill_w * phase)
        scan_ww    = max(3, seg_w // 10)
        hi = (min(255, color[0]+60), min(255, color[1]+60), min(255, color[2]+60), min(255, color[3]+10))
        rr(scan_x, by0+2, min(sx0+fill_w, scan_x+scan_ww), by0+band_h-2, 6, hi)

    draw_seg(0, cpu_u, band_cpu)
    draw_seg(1, ram_u, band_ram)
    draw_seg(2, gpu_u, band_gpu)

    cx = int((hx0 + hx1) * 0.5)
    cy = by0 + int(band_h * 0.5)
    r  = max(8, int(w*0.018))
    rg = int(r * (1.8 + 1.2*net_u))
    d.polygon([(cx, cy-rg),(cx+rg, cy),(cx, cy+rg),(cx-rg, cy)], fill=diamond_glow)
    d.polygon([(cx, cy-r), (cx+r,  cy),(cx, cy+r), (cx-r,  cy)], fill=diamond_base)

    rr(int(w*0.16), int(h*0.26), int(w*0.84), int(h*0.96), 42, cloak_c)
    rr(int(w*0.36), int(h*0.33), int(w*0.64), int(h*0.78), 34, body_c)
    rr(int(w*0.40), int(h*0.78), int(w*0.60), int(h*0.95), 26, legs_c)
    rr(int(w*0.22), int(h*0.40), int(w*0.36), int(h*0.72), 26, arms_c)
    rr(int(w*0.64), int(h*0.40), int(w*0.78), int(h*0.72), 26, arms_c)
    rr(int(w*0.20), int(h*0.70), int(w*0.30), int(h*0.80), 18, hands_c)
    rr(int(w*0.70), int(h*0.70), int(w*0.80), int(h*0.80), 18, hands_c)
    rr(int(w*0.44), int(h*0.60), int(w*0.47), int(h*0.86), 10, tassel_c)
    rr(int(w*0.53), int(h*0.60), int(w*0.56), int(h*0.86), 10, tassel_c)

    return _apply_simple_lighting(im, ambient, strength)

# endregion

# region ui components

class BubbleToast(QFrame):
    def __init__(self, title: str, text: str, parent=None):
        super().__init__(parent)
        self.setObjectName("BubbleToast")
        self.setStyleSheet(
            "QFrame#BubbleToast { background: rgba(10,12,18,0.86); border: 1px solid rgba(148,163,184,0.22); border-radius: 10px; } "
            "QLabel { color: rgba(226,232,240,0.92); }"
        )
        lay = QVBoxLayout(self); lay.setContentsMargins(10,8,10,8); lay.setSpacing(4)
        hdr = QLabel(title)
        hdr.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        hdr.setStyleSheet("color: rgba(167,139,250,0.90);")
        lay.addWidget(hdr)
        body = QLabel(text); body.setWordWrap(True)
        body.setFont(QFont("Consolas", 9))
        body.setStyleSheet("color: rgba(226,232,240,0.88);")
        lay.addWidget(body)
        self._eff = QGraphicsOpacityEffect(self)
        self._eff.setOpacity(1.0)
        self.setGraphicsEffect(self._eff)

    def fade_out_and_delete(self, ms: int = 700):
        anim = QPropertyAnimation(self._eff, b"opacity", self)
        anim.setDuration(ms); anim.setStartValue(1.0); anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.finished.connect(self.deleteLater)
        anim.start(); self._anim = anim

class BubbleStack(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self._bubbles: list[BubbleToast] = []
        self.max_visible        = 6
        self.gap                = 8
        self.entry_margin_bottom= 10
        self.entry_margin_right = 10

    def add_bubble(self, bubble: BubbleToast, ttl_ms: int):
        self._bubbles = [b for b in self._bubbles if isValid(b)]
        bubble.setParent(self); bubble.show()
        if len(self._bubbles) >= self.max_visible:
            old = self._bubbles.pop(0)
            if isValid(old):
                old.fade_out_and_delete(ms=350)
        bubble.adjustSize()
        dy = bubble.sizeHint().height() + self.gap
        for b in list(self._bubbles):
            if isValid(b): b.move(b.x(), b.y() - dy)
        x = max(0, self.width()  - bubble.sizeHint().width()  - self.entry_margin_right)
        y = max(0, self.height() - bubble.sizeHint().height() - self.entry_margin_bottom)
        bubble.move(x, y); self._bubbles.append(bubble)
        def _expire():
            self._bubbles = [b for b in self._bubbles if isValid(b) and b is not bubble]
            if isValid(bubble): bubble.fade_out_and_delete()
        QTimer.singleShot(ttl_ms, _expire)

class SnapshotToastWindow(QWidget):
    """
    Brief translucent flash of the captured screenshot.
    Visible  process: fades out after 3.5s (normal, blue border)
    Hidden   process: fades out after 6s, red border + warning label
    """
    def __init__(self, pixmap: QPixmap, ttl_ms: int = 3500, hiddenProc: bool = False,
                 procName: str = "", max_size: QSize = None):
        super().__init__()
        if max_size is None:
            max_size = QSize(480, 300)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(4)

        self.imgLabel = QLabel()
        self.imgLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        borderCol = "rgba(248,113,113,0.70)" if hiddenProc else "rgba(148,163,184,0.22)"
        self.imgLabel.setStyleSheet(f"""
            QLabel {{
                background: rgba(10, 12, 18, 0.86);
                border: 1px solid {borderCol};
                border-radius: 10px;
                padding: 5px;
            }}
        """)
        scaled = pixmap.scaled(max_size, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        self.imgLabel.setPixmap(scaled)
        root.addWidget(self.imgLabel)

        if hiddenProc:
            warn = QLabel(f"\u26a0  not visible: {procName}")
            warn.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
            warn.setStyleSheet("color: rgba(248,113,113,0.90);")
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(warn)

        self.adjustSize()
        self._eff = QGraphicsOpacityEffect(self)
        self._eff.setOpacity(0.0)
        self.setGraphicsEffect(self._eff)

        fadeIn = QPropertyAnimation(self._eff, b"opacity", self)
        fadeIn.setDuration(250)
        fadeIn.setStartValue(0.0)
        fadeIn.setEndValue(1.0)
        fadeIn.setEasingCurve(QEasingCurve.Type.OutQuad)
        fadeIn.start()
        self._fadeIn = fadeIn
        self._fadeOut = None
        QTimer.singleShot(ttl_ms, self._startFadeOut)

    def _startFadeOut(self):
        if self._fadeOut is not None:
            return
        anim = QPropertyAnimation(self._eff, b"opacity", self)
        anim.setDuration(700)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.finished.connect(self.close)
        self._fadeOut = anim
        anim.start()

class NoteParticle(QLabel):
    def __init__(self, note_char: str, parent: QWidget):
        super().__init__(note_char, parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        self.setStyleSheet("color: rgba(180, 220, 255, 0.85);")
        self._eff = QGraphicsOpacityEffect(self)
        self._eff.setOpacity(1.0)
        self.setGraphicsEffect(self._eff)

    def launch(self, start: QPoint):
        self.move(start); self.show()
        ap = QPropertyAnimation(self, b"pos", self)
        ap.setDuration(1200); ap.setStartValue(start)
        ap.setEndValue(QPoint(start.x(), start.y() - 90))
        ap.setEasingCurve(QEasingCurve.Type.OutQuad)
        ao = QPropertyAnimation(self._eff, b"opacity", self)
        ao.setDuration(1200); ao.setStartValue(1.0); ao.setEndValue(0.0)
        ao.setEasingCurve(QEasingCurve.Type.InOutQuad)
        ao.finished.connect(self.deleteLater)
        self._ap = ap; self._ao = ao; ap.start(); ao.start()

# endregion

# region avatar window

class AvatarWindow(QWidget):
    def __init__(self, always_on_top: bool = True):
        super().__init__()
        flags = Qt.FramelessWindowHint | Qt.Tool
        if always_on_top:
            flags |= Qt.WindowStaysOnTopHint
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("SBS_Avatar")
        self.resize(420, 520)
        self.notes_enabled      = False
        self._torso_flash_until = 0.0

        root = QVBoxLayout(self); root.setContentsMargins(10,10,10,10); root.setSpacing(8)
        self.avatar_label = QLabel()
        self.avatar_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar_label.setMinimumSize(QSize(320, 320))
        self.avatar_label.setStyleSheet(
            "QLabel { background: rgba(2,6,23,0.00); border: 1px solid rgba(148,163,184,0.16); "
            "border-radius: 14px; padding: 8px; }"
        )
        root.addWidget(self.avatar_label, 1)

        hint = QLabel("left-drag move  •  right-drag resize  •  ESC closes SBS")
        hint.setFont(QFont("Arial", 8))
        hint.setStyleSheet("color: rgba(148,163,184,0.70);")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)

        self.bubbles = BubbleStack(self.avatar_label)
        self.bubbles.setGeometry(self.avatar_label.rect())
        self.bubbles.show()

        self._note_timer = QTimer(self)
        self._note_timer.timeout.connect(self._maybe_note)
        self._note_timer.start(300)

        self._drag_pos         = None
        self._resizing         = False
        self._resize_start_pos = None
        self._resize_start_geo = None

    def torso_flash(self, seconds: float = 0.45):
        self._torso_flash_until = max(self._torso_flash_until, time.time() + seconds)

    def torso_flash_level(self) -> float:
        rem = self._torso_flash_until - time.time()
        return clamp(rem / 0.45) if rem > 0 else 0.0

    def set_avatar_image(self, im: Image.Image):
        buf  = io.BytesIO(); im.save(buf, format="PNG")
        qimg = QImage.fromData(buf.getvalue())
        px   = QPixmap.fromImage(qimg)
        self.avatar_label.setPixmap(
            px.scaled(self.avatar_label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                      Qt.TransformationMode.SmoothTransformation)
        )

    def add_bubble(self, title: str, text: str, ttl_ms: int):
        self.bubbles.add_bubble(BubbleToast(title, text, parent=self.bubbles), ttl_ms=ttl_ms)

    def music_burst(self, n: int = 3):
        for _ in range(n): self._spawn_note()

    def _maybe_note(self):
        if self.notes_enabled and random.random() < 0.16:
            self._spawn_note()

    def _spawn_note(self):
        p = NoteParticle(random.choice(["♪","♫","♩","♬"]), parent=self.avatar_label)
        w = self.avatar_label.width(); h = self.avatar_label.height()
        p.launch(QPoint(int(w*0.52), int(h*0.30)))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._resizing         = True
            self._resize_start_pos = event.globalPosition().toPoint()
            self._resize_start_geo = self.geometry()
            event.accept()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos); event.accept()
        elif (event.buttons() & Qt.MouseButton.RightButton) and self._resizing:
            delta = event.globalPosition().toPoint() - self._resize_start_pos
            self.resize(max(300, self._resize_start_geo.width()  + delta.x()),
                        max(360, self._resize_start_geo.height() + delta.y()))
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = None
        elif event.button() == Qt.MouseButton.RightButton:
            self._resizing = False; self._resize_start_pos = None; self._resize_start_geo = None
        event.accept()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.bubbles.setGeometry(self.avatar_label.rect())

# endregion

# region main application

class SBS(QWidget):
    def __init__(self):
        super().__init__()
        self.config  = ConfigManager(CONFIG_PATH)
        self.journal = JournalManager(JOURNAL_PATH)
        self.actions = ActionLog(ACTIONS_PATH)
        self.typingbio = TypingBioLog(TYPING_PATH)
        self.memory  = LanguageMemory(MEMORY_PATH)
        self.key     = read_or_create_key(KEY_PATH)

        self.bus = EventBus()
        self.bus.subscribe_any(lambda e: self.actions.write(e), priority=0)
        self.bus.subscribe("machine_event",  self._on_machine,        priority=50)
        self.bus.subscribe("media_event",    self._on_media,          priority=50)
        self.bus.subscribe("snapshot_event", self._on_snapshot_event, priority=50)
        self.bus.subscribe("proc_event",     self._on_proc_event,     priority=50)
        self.bus.subscribe("focus_event",    self._on_focus_event,    priority=50)

        self.stability = StabilityModel()
        self.net       = NetObserver()
        self.gate      = EventGate()

        self.procmon  = ProcessRosterMonitor(self.bus, every_s=int(self.config.get("processRosterEveryS", 20)))
        self.focusmon = WindowFocusMonitor(self.bus,  poll_ms=int(self.config.get("focusPollMs", 400)))

        self.is_bootstrapping = not self.config.is_initialized()
        self.bootstrap_step   = 0
        self.shepherd         = self.config.get("shepherdName")
        self.system_name      = self.config.get("systemName")

        self.avatar      = AvatarWindow(always_on_top=bool(self.config.get("avatarAlwaysOnTop", True)))
        self.media_window= None

        self._glow_until    = 0.0
        self._last_mode     = "A"
        self._cpu           = 0.0
        self._ram           = 0.0
        self._gpu           = 0.0
        self._disk          = 0.0
        self._net_in        = 0.0
        self._net_out       = 0.0
        self._cpu_freq_ratio= 0.0

        self.snapshot_cooldown_s    = int(self.config.get("snapshotCooldownS", 60))
        self.snapshot_next_allowed  = 0.0
        self.last_snapshot_record   = None
        self.tamper_alarm           = False
        self.mismatch_alarm_until   = 0.0
        self._procSnapTimes: dict[str, float] = {}   # per-process last-snapshot epoch
        self._activeToasts:  list             = []   # live SnapshotToastWindow refs

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("ShepherdBootStrap")
        self.resize(900, 600)

        self._drag_pos         = None
        self._resizing         = False
        self._resize_start_pos = None
        self._resize_start_geo = None

        root = QVBoxLayout(self); root.setContentsMargins(12,12,12,12); root.setSpacing(10)

        self.title = QLabel("◼ System" if self.is_bootstrapping else f"◼ {self.system_name}")
        self.title.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        self.title.setStyleSheet("color: rgba(167,139,250,0.85);")
        root.addWidget(self.title)

        self.input = BioLineEdit(self.typingbio)
        self.input.setFont(QFont("Consolas", 11))
        self.input.setFixedHeight(44)
        self.input.returnPressed.connect(self._on_send)
        self.input.setStyleSheet(
            "QLineEdit { background: rgba(10,12,18,0.78); color: rgba(226,232,240,0.95); "
            "border: 1px solid rgba(148,163,184,0.22); border-radius: 10px; "
            "padding: 10px 12px; font-size: 16px; }"
        )
        root.addWidget(self.input)

        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setFont(QFont("Consolas", 10))
        self.display.setFrameStyle(QTextEdit.NoFrame)
        self.display.setStyleSheet(
            "QTextEdit { background: rgba(2,6,23,0.06); color: rgba(226,232,240,0.72); "
            "border: 1px solid rgba(148,163,184,0.10); border-radius: 10px; padding: 10px; }"
        )
        root.addWidget(self.display, 1)

        if self.is_bootstrapping:
            self._append("sys", "System requires initialization. What is your name?")
            self.input.setPlaceholderText("type your name and press Enter...")
        else:
            self._append("sys", f"Welcome back, {self.shepherd}.")
            self._append("sys", f"Memory: {self.memory.stats()}")
            self.input.setPlaceholderText("type something...")

        self.bus.emit(make_event("session", "SBS", event="start", build="SBS_FULL_ONEFILE"))
        # session boundary goes to actions only, not journal

        sample_ms = int(self.config.get("resourceSampleMs", 2000))
        anim_ms   = int(self.config.get("avatarAnimMs", 150))
        focus_ms  = int(self.config.get("focusPollMs", 400))

        self._timer = QTimer(self); self._timer.timeout.connect(self._tick); self._timer.start(sample_ms)
        self._avatar_timer = QTimer(self); self._avatar_timer.timeout.connect(self._avatar_tick); self._avatar_timer.start(anim_ms)
        self._focus_timer  = QTimer(self); self._focus_timer.timeout.connect(self.focusmon.tick);  self._focus_timer.start(focus_ms)

        self.avatar.show()
        self.show(); self.raise_(); self.activateWindow(); self.move(120, 120)
        self.avatar.move(self.x() + self.width() + 18, self.y() + 18)

    # ── window drag / resize ─────────────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.RightButton:
            self._resizing         = True
            self._resize_start_pos = event.globalPosition().toPoint()
            self._resize_start_geo = self.geometry()
            event.accept()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.LeftButton) and self._drag_pos:
            self.move(event.globalPosition().toPoint() - self._drag_pos); event.accept()
        elif (event.buttons() & Qt.RightButton) and self._resizing:
            delta = event.globalPosition().toPoint() - self._resize_start_pos
            self.resize(max(520, self._resize_start_geo.width()  + delta.x()),
                        max(360, self._resize_start_geo.height() + delta.y()))
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_pos = None
        elif event.button() == Qt.RightButton:
            self._resizing = False; self._resize_start_pos = None; self._resize_start_geo = None
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close(); return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.bus.emit(make_event("session", "SBS", event="end"))
        # session end goes to actions log only
        try:
            if self.media_window is not None and isValid(self.media_window):
                self.media_window.close()
        except Exception:
            pass
        try:
            if self.avatar is not None and isValid(self.avatar):
                self.avatar.close()
        except Exception:
            pass
        event.accept()

    # ── display helpers ──────────────────────────────────────────────────────

    def _append(self, role: str, text: str):
        ts = now_hms()
        self.display.append(f"[{ts}] {role} > {esc_html(text)}")
        self.display.moveCursor(QTextCursor.End)
        self.display.ensureCursorVisible()

    def _sys(self, text: str):
        """System reply — shown in display and recorded in journal as chat."""
        self._append("sys", text)
        self.journal.write_chat("sys", text)

    # ── event handlers ───────────────────────────────────────────────────────

    def _on_machine(self, e: Event):
        d   = e["data"]
        ttl = int(self.config.get("avatarBubbleTtlMs", 5000))
        self.avatar.torso_flash(0.45)
        self.avatar.add_bubble(f'{d.get("subsystem","SYS")} • {e.get("mode","")}', d.get("msg",""), ttl_ms=ttl)
        self._glow_until = time.time() + 1.5

    def _on_media(self, e: Event):
        d      = e["data"]
        action = d.get("action", "")
        if action == "play":
            self.avatar.notes_enabled = True
        elif action in ("pause", "stop"):
            self.avatar.notes_enabled = False
        elif action == "track_change":
            self.avatar.notes_enabled = True
            self.avatar.music_burst(3)

    def _on_snapshot_event(self, e: Event):
        self.avatar.torso_flash(0.25)

    def _on_proc_event(self, e: Event):
        d = e["data"]
        if d.get("action") == "new" and bool(self.config.get("snapshotEnabled", True)):
            procName = d.get("process", "")
            if procName:
                self._attempt_snapshot(reason="new_process", procName=procName)

    def _on_focus_event(self, e: Event):
        pass  # logged to WindowFocus.jsonl and system_actions.jsonl only

    # ── snapshot + integrity ─────────────────────────────────────────────────

    def _attempt_snapshot(self, reason: str, procName: str = "") -> None:
        """
        Fire a snapshot. When triggered by a new process:
        - Cross-references visible windows to determine if procName is on screen
        - Hidden process  → red-bordered toast, longer hand glow, orange diamond pulse
        - Visible process → normal blue-bordered toast, brief hand glow
        Per-process cooldown prevents duplicate snapshots for the same process name.
        Global snapshotEnabled toggle silences toasts but still saves the file.
        """
        now = time.time()

        # per-process cooldown — same process won't re-trigger within cooldown window
        if procName:
            lastSnap = self._procSnapTimes.get(procName, 0.0)
            if now - lastSnap < self.snapshot_cooldown_s:
                return
            self._procSnapTimes[procName] = now
        else:
            remaining = max(0.0, self.snapshot_next_allowed - now)
            if remaining > 0:
                self.bus.emit(make_event("snapshot_event", "SBS", event="blocked",
                                         reason=reason, cooldown_remaining_s=round(remaining, 2)))
                return

        screen = QApplication.primaryScreen()
        if screen is None:
            self.bus.emit(make_event("snapshot_event", "SBS", event="error", reason="no_screen"))
            return

        px   = screen.grabWindow(0)
        qimg = px.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        ptr  = qimg.bits()
        pil  = Image.frombuffer("RGBA", (w, h), ptr, "raw", "RGBA", 0, 1)

        ts       = datetime.now().strftime("%H%M%S")
        fname    = f"snap_{datetime.now().strftime('%Y%m%d')}_{ts}.png"
        outPath  = SNAP_DIR / fname
        buf      = io.BytesIO()
        pil.save(buf, format="PNG")
        pngBytes = buf.getvalue()
        outPath.write_bytes(pngBytes)

        sha  = sha256_bytes(pngBytes)
        ph   = dhash_hex_from_pil(pil)
        meta = get_visible_windows_metadata()
        screenMeta = {"w": w, "h": h, "monitors": 1}
        msg  = json.dumps({"sha": sha, "ph": ph, "fg": meta.get("foreground"), "sz": screenMeta},
                          sort_keys=True).encode("utf-8")
        sig  = hmac_hex(self.key, msg)

        # visibility cross-reference
        visibleNames = {v.get("process", "").lower() for v in meta.get("visible", [])}
        hiddenProc   = bool(procName and procName.lower() not in visibleNames)

        record = {
            "event":      "taken",
            "reason":     reason,
            "proc":       procName,
            "hidden":     hiddenProc,
            "path":       str(outPath.relative_to(ROOT)),
            "sha256":     sha,
            "phash":      ph,
            "hmac":       sig,
            "screen":     screenMeta,
            "windows":    meta,
        }
        self.last_snapshot_record  = record
        self.snapshot_next_allowed = now + self.snapshot_cooldown_s
        self.bus.emit(make_event("snapshot_event", "SBS", **record))

        # avatar signals
        if hiddenProc:
            # hands glow longer, mismatch-style orange pulse on diamond
            self.avatar.torso_flash(1.2)
            self.mismatch_alarm_until = max(self.mismatch_alarm_until, now + 4.0)
            ttl = 6000
            self.avatar.add_bubble(
                f"⚠ hidden process",
                f"{procName} is running but not visible on screen",
                ttl_ms=ttl,
            )
        else:
            self.avatar.torso_flash(0.45)
            ttl = 3500

        # toast — shown regardless of hidden/visible, can be toggled off
        if bool(self.config.get("snapshotEnabled", True)):
            toast = SnapshotToastWindow(px, ttl_ms=ttl, hiddenProc=hiddenProc, procName=procName)
            # position to the left of avatar
            ax = self.avatar.x()
            ay = self.avatar.y()
            toast.adjustSize()
            toast.move(max(0, ax - toast.width() - 12), ay + 12)
            toast.show()
            self._activeToasts.append(toast)
            # prune dead toasts
            self._activeToasts = [t for t in self._activeToasts if isValid(t)]

    def _verify_last_snapshot_integrity(self) -> tuple[bool, bool]:
        if not self.last_snapshot_record:
            return (False, False)
        rec = self.last_snapshot_record
        try:
            p = ROOT / rec["path"]
            if not p.exists():
                return (True, False)
            png_bytes = p.read_bytes()
            if sha256_bytes(png_bytes) != rec["sha256"]:
                return (True, False)
            msg = json.dumps({"sha": rec["sha256"], "ph": rec["phash"],
                              "fg": rec["windows"].get("foreground"), "sz": rec["screen"]},
                             sort_keys=True).encode("utf-8")
            if not hmac_ok(self.key, msg, rec["hmac"]):
                return (True, False)
        except Exception:
            return (True, False)
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                return (False, False)
            px   = screen.grabWindow(0)
            qimg = px.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            ptr  = qimg.bits()
            pil  = Image.frombuffer("RGBA", (w, h), ptr, "raw", "RGBA", 0, 1)
            cur_ph  = dhash_hex_from_pil(pil)
            dist    = hamming64_hex(cur_ph, rec["phash"])
            mismatch= dist > 18
            return (False, mismatch)
        except Exception:
            return (False, False)

    # ── input handler ────────────────────────────────────────────────────────

    def _on_send(self):
        raw = (self.input.text() or "").strip()
        if not raw:
            return
        self.input.clear()

        # record typing biometrics to typingbio.jsonl
        bio = self.typingbio.on_submit(raw)
        self.bus.emit(make_event("typingbio", "SBS", **bio))

        clean, cmds = parse_commands(raw)
        for c in cmds:
            if c["domain"] == "media" and c["action"] == "player":
                self.bus.emit(make_event("media_event", "SBS", action="play", msg="media state ON"))
            elif c["domain"] == "close" and c["action"] == "media":
                self.bus.emit(make_event("media_event", "SBS", action="stop", msg="media state OFF"))
        if not clean:
            return

        # display + journal (chat only)
        self._append("user", clean)
        self.journal.write_chat(self.shepherd or "user", clean)

        # bootstrap flow
        if self.is_bootstrapping:
            if self.bootstrap_step == 0:
                self.shepherd       = clean
                self.bootstrap_step = 1
                self._sys("Welcome. What will you name the system?")
                self.input.setPlaceholderText("type system name and press Enter...")
                return
            if self.bootstrap_step == 1:
                self.system_name = clean
                self.config.initialize(self.shepherd, self.system_name)
                self.is_bootstrapping = False
                self.title.setText(f"◼ {self.system_name}")
                self._sys("Initialization complete.")
                self._sys(f"Memory: {self.memory.stats()}")
                self.input.setPlaceholderText("type something...")
                return

        # language memory
        reply, conf, matched, top_tokens = self.memory.best_match(clean)
        if matched is None:
            self.bus.emit(make_event("mimic", "SBS", confidence=round(conf, 4), top_tokens=top_tokens))
            self.memory.add_pair(clean, clean, event="mimic")
            self._sys(clean)
            return
        self.bus.emit(make_event("match", "SBS", confidence=round(conf, 4),
                                  matched_user=matched.user[:120], top_tokens=top_tokens))
        self.memory.add_pair(clean, reply, event="pair")
        self._sys(reply)

    # ── telemetry tick ───────────────────────────────────────────────────────

    def _tick(self):
        self.procmon.tick()
        self._cpu  = psutil.cpu_percent(interval=None)
        self._ram  = psutil.virtual_memory().percent
        self._gpu  = get_gpu_usage_ratio()
        try:
            self._disk = psutil.disk_usage(str(ROOT)).percent
        except Exception:
            self._disk = 0.0
        self._net_in, self._net_out = self.net.sample()
        self._cpu_freq_ratio        = get_cpu_freq_ratio()
        self._last_mode             = self.stability.compute(self._cpu, self._ram)
        tag = mode_tag_from_letter(self._last_mode)
        for msg, sub in (
            (self.gate.cpu_event(self._cpu),                   "CPU"),
            (self.gate.ram_event(self._ram),                   "RAM"),
            (self.gate.disk_event(self._disk),                 "DISK"),
            (self.gate.net_event(self._net_in, self._net_out), "NET"),
        ):
            if msg:
                self.bus.emit(make_event("machine_event", "SBS", mode=tag, subsystem=sub, msg=msg))
        tampered, mismatch = self._verify_last_snapshot_integrity()
        self.tamper_alarm  = tampered
        if mismatch and not tampered:
            self.mismatch_alarm_until = max(self.mismatch_alarm_until, time.time() + 2.5)

    # ── avatar tick ──────────────────────────────────────────────────────────

    def _apply_alarm_overlays(self, img: Image.Image, diamond_state: str, hands_glow: float) -> Image.Image:
        w, h = img.size
        d    = ImageDraw.Draw(img, "RGBA")
        head_w, head_h = int(w*0.24), int(h*0.20)
        hx0, hy0 = int(w*0.38), int(h*0.08)
        hx1      = hx0 + head_w
        band_h   = max(10, int(h*0.03))
        by0      = hy0 + int(head_h*0.32)
        cx       = int((hx0 + hx1) * 0.5)
        cy       = by0 + int(band_h * 0.5)

        if diamond_state == "tamper":
            phase = math.sin(time.time() * 6.0) * 0.5 + 0.5
            col   = (255, 30, 30, 220) if phase > 0.5 else (0, 0, 0, 220)
            r     = max(10, int(w*0.020))
            d.polygon([(cx, cy-r),(cx+r, cy),(cx, cy+r),(cx-r, cy)], fill=col)
        elif diamond_state == "mismatch":
            pulse = math.sin(time.time() * 3.3) * 0.5 + 0.5
            col   = (255, 140, 0, int(80 + 150*pulse))
            r     = max(10, int(w*0.020))
            d.polygon([(cx, cy-r),(cx+r, cy),(cx, cy+r),(cx-r, cy)], fill=col)

        if hands_glow > 0:
            a   = int(30 + 180 * hands_glow)
            col = (140, 190, 255, a)
            d.rounded_rectangle([int(w*0.20), int(h*0.70), int(w*0.30), int(h*0.80)], radius=18, fill=col)
            d.rounded_rectangle([int(w*0.70), int(h*0.70), int(w*0.80), int(h*0.80)], radius=18, fill=col)
        return img

    def _avatar_tick(self):
        now             = time.time()
        remaining       = max(0.0, self.snapshot_next_allowed - now)
        hands_glow      = clamp(remaining / max(1.0, float(self.snapshot_cooldown_s)))
        mismatch_active = (time.time() < self.mismatch_alarm_until)
        diamond_state   = "tamper" if self.tamper_alarm else ("mismatch" if mismatch_active else "normal")
        img = render_avatar_telemetry(
            w=360, h=360,
            cpu=self._cpu, ram=self._ram, gpu=self._gpu, disk=self._disk,
            net_in_kbps=self._net_in, net_out_kbps=self._net_out,
            cpu_freq_ratio=self._cpu_freq_ratio,
            torso_flash=self.avatar.torso_flash_level(),
            mode=self._last_mode,
        )
        img = self._apply_alarm_overlays(img, diamond_state, hands_glow)
        self.avatar.set_avatar_image(img)

# endregion

# region entry point

def main():
    app = QApplication(sys.argv)
    try:
        app.setFont(QFont("Consolas", 10))
    except Exception:
        pass

    config = ConfigManager(CONFIG_PATH)
    termsAccepted = config.data.get("termsAccepted", False)

    if not termsAccepted:
        terms = TermsWindow()

        def onAccepted():
            config.data["termsAccepted"] = True
            config.save()
            terms.deleteLater()
            mainWindow = SBS()  # noqa: F841  kept alive by Qt

        terms.accepted.connect(onAccepted)
        terms.show()
    else:
        SBS()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
# endregion