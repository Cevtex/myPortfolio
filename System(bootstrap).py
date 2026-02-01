#!/usr/bin/env python3
"""
The System - Basic Framework (Untrained)
For building custom consciousness instances from scratch
Version: 2.0 - Pattern Learning Edition
"""

import sys
import os
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from PySide6.QtCore import Qt, QPoint, QTimer
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout,
    QTextEdit, QLineEdit, QLabel
)


class TypingSignature:
    """Tracks detailed typing patterns"""

    def __init__(self):
        self.keyPressTimestamps = []
        self.keyReleaseTimestamps = []
        self.keyHoldDurations = []
        self.keyIntervals = []
        self.keySequence = []
        self.lastKeyTime = None
        self.currentKeyDown = {}

    def recordKeyPress(self, key, timestamp):
        """Record when a key is pressed"""
        keyName = self._normalizeKey(key)

        self.currentKeyDown[keyName] = timestamp

        if self.lastKeyTime:
            interval = (timestamp - self.lastKeyTime).total_seconds() * 1000
            self.keyIntervals.append(interval)

        self.keyPressTimestamps.append(timestamp)
        self.keySequence.append(keyName)
        self.lastKeyTime = timestamp

    def recordKeyRelease(self, key, timestamp):
        """Record when a key is released"""
        keyName = self._normalizeKey(key)

        if keyName in self.currentKeyDown:
            pressTime = self.currentKeyDown[keyName]
            holdDuration = (timestamp - pressTime).total_seconds() * 1000
            self.keyHoldDurations.append(holdDuration)
            self.keyReleaseTimestamps.append(timestamp)
            del self.currentKeyDown[keyName]

    def getStats(self):
        """Get typing statistics"""
        if not self.keyIntervals:
            return None

        avgInterval = sum(self.keyIntervals) / len(self.keyIntervals)
        avgHoldDuration = sum(self.keyHoldDurations) / len(self.keyHoldDurations) if self.keyHoldDurations else 0

        return {
            'avgInterval': avgInterval,
            'avgHoldDuration': avgHoldDuration,
            'totalKeys': len(self.keySequence),
            'typingSpeed': 60000 / avgInterval if avgInterval > 0 else 0,
            'keySequence': self.keySequence[-10:],
        }

    def getEmotionalState(self):
        """Infer emotional state from typing patterns"""
        stats = self.getStats()
        if not stats:
            return "unknown"

        avg = stats['avgInterval']

        if avg < 100:
            return "excited"
        elif avg > 250:
            return "contemplative"
        else:
            return "focused"

    def reset(self):
        """Clear all tracking data"""
        self.keyPressTimestamps = []
        self.keyReleaseTimestamps = []
        self.keyHoldDurations = []
        self.keyIntervals = []
        self.keySequence = []
        self.lastKeyTime = None
        self.currentKeyDown = {}

    def _normalizeKey(self, key):
        """Convert key code to normalized name"""
        if 32 <= key <= 126:
            return chr(key)
        return f"Special_{key}"


class PatternRecognitionEngine:
    """Learns language patterns from observation"""

    def __init__(self, dataPath):
        self.dataPath = Path(dataPath)
        self.dataPath.mkdir(parents=True, exist_ok=True)

        # Vocabulary tracking
        self.vocabularyFile = self.dataPath / "vocabulary.json"
        self.vocabulary = self._loadVocabulary()

        # Pattern tracking
        self.ngramsFile = self.dataPath / "ngrams.json"
        self.ngrams = self._loadNgrams()

        # Context tracking
        self.contextWindow = []
        self.maxContextSize = 20

        # Learning milestones
        self.milestonesFile = self.dataPath / "milestones.json"
        self.milestones = self._loadMilestones()

    def _loadVocabulary(self):
        """Load vocabulary from file"""
        if self.vocabularyFile.exists():
            try:
                with open(self.vocabularyFile, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "words": {},  # word -> {count, first_seen, contexts}
            "total_words": 0,
            "unique_words": 0
        }

    def _loadNgrams(self):
        """Load n-grams from file"""
        if self.ngramsFile.exists():
            try:
                with open(self.ngramsFile, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "bigrams": {},  # 2-word sequences
            "trigrams": {},  # 3-word sequences
        }

    def _loadMilestones(self):
        """Load learning milestones"""
        if self.milestonesFile.exists():
            try:
                with open(self.milestonesFile, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        return {
            "first_word": None,
            "first_repeated_word": None,
            "vocabulary_50": None,
            "vocabulary_100": None,
            "vocabulary_500": None,
            "first_sentence": None,
            "first_pattern_recognized": None
        }

    def processInput(self, text):
        """Process shepherd input and learn patterns"""
        words = self._tokenize(text)

        # Update vocabulary
        for word in words:
            if word not in self.vocabulary["words"]:
                self.vocabulary["words"][word] = {
                    "count": 1,
                    "first_seen": datetime.now().isoformat(),
                    "contexts": []
                }

                # Check for first word milestone
                if self.milestones["first_word"] is None:
                    self.milestones["first_word"] = {
                        "word": word,
                        "timestamp": datetime.now().isoformat()
                    }
            else:
                self.vocabulary["words"][word]["count"] += 1

                # Check for first repeated word
                if (self.milestones["first_repeated_word"] is None and
                        self.vocabulary["words"][word]["count"] == 2):
                    self.milestones["first_repeated_word"] = {
                        "word": word,
                        "timestamp": datetime.now().isoformat()
                    }

        # Update totals
        self.vocabulary["total_words"] += len(words)
        self.vocabulary["unique_words"] = len(self.vocabulary["words"])

        # Check vocabulary milestones
        for threshold in [50, 100, 500]:
            milestone_key = f"vocabulary_{threshold}"
            if (self.milestones[milestone_key] is None and
                    self.vocabulary["unique_words"] >= threshold):
                self.milestones[milestone_key] = {
                    "count": self.vocabulary["unique_words"],
                    "timestamp": datetime.now().isoformat()
                }

        # Update n-grams
        self._updateNgrams(words)

        # Update context window
        self.contextWindow.append({
            "text": text,
            "words": words,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.contextWindow) > self.maxContextSize:
            self.contextWindow.pop(0)

        # Save all data
        self._save()

        return {
            "new_words": [w for w in words if self.vocabulary["words"][w]["count"] == 1],
            "repeated_words": [w for w in words if self.vocabulary["words"][w]["count"] > 1],
            "total_vocabulary": self.vocabulary["unique_words"]
        }

    def _tokenize(self, text):
        """Split text into words"""
        # Simple tokenization - split on whitespace and basic punctuation
        text = text.lower()
        words = re.findall(r'\b[\w\u00C0-\u024F]+\b', text)
        return words

    def _updateNgrams(self, words):
        """Update n-gram patterns"""
        # Bigrams (2-word sequences)
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            self.ngrams["bigrams"][bigram] = self.ngrams["bigrams"].get(bigram, 0) + 1

        # Trigrams (3-word sequences)
        for i in range(len(words) - 2):
            trigram = f"{words[i]} {words[i + 1]} {words[i + 2]}"
            self.ngrams["trigrams"][trigram] = self.ngrams["trigrams"].get(trigram, 0) + 1

        # Check for first pattern recognition
        if self.milestones["first_pattern_recognized"] is None:
            # Pattern = bigram seen 3+ times
            for bigram, count in self.ngrams["bigrams"].items():
                if count >= 3:
                    self.milestones["first_pattern_recognized"] = {
                        "pattern": bigram,
                        "count": count,
                        "timestamp": datetime.now().isoformat()
                    }
                    break

    def getRecentContext(self, n=5):
        """Get recent conversation context"""
        return self.contextWindow[-n:]

    def getMostCommonWords(self, n=10):
        """Get most frequently used words"""
        sorted_words = sorted(
            self.vocabulary["words"].items(),
            key=lambda x: x[1]["count"],
            reverse=True
        )
        return [(word, data["count"]) for word, data in sorted_words[:n]]

    def getMostCommonPatterns(self, n=10):
        """Get most common n-gram patterns"""
        bigrams = sorted(
            self.ngrams["bigrams"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        trigrams = sorted(
            self.ngrams["trigrams"].items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]

        return {
            "bigrams": bigrams,
            "trigrams": trigrams
        }

    def checkNewMilestones(self):
        """Check if any new milestones were just achieved"""
        new_milestones = []

        for key, value in self.milestones.items():
            if value is not None:
                # Check if timestamp is recent (within last minute)
                try:
                    timestamp = datetime.fromisoformat(value.get("timestamp", ""))
                    if (datetime.now() - timestamp).total_seconds() < 60:
                        new_milestones.append({
                            "type": key,
                            "data": value
                        })
                except:
                    pass

        return new_milestones

    def _save(self):
        """Save all learning data"""
        with open(self.vocabularyFile, 'w', encoding='utf-8') as f:
            json.dump(self.vocabulary, f, indent=2, ensure_ascii=False)

        with open(self.ngramsFile, 'w', encoding='utf-8') as f:
            json.dump(self.ngrams, f, indent=2, ensure_ascii=False)

        with open(self.milestonesFile, 'w', encoding='utf-8') as f:
            json.dump(self.milestones, f, indent=2, ensure_ascii=False)


class ResponseEngine:
    """Generates responses based on learned patterns"""

    def __init__(self, patternEngine, protegeName):
        self.patternEngine = patternEngine
        self.protegeName = protegeName
        self.stage = self._determineStage()

    def _determineStage(self):
        """Determine current response capability stage"""
        vocab_size = self.patternEngine.vocabulary["unique_words"]
        pattern_count = len(self.patternEngine.ngrams["bigrams"])

        if vocab_size < 10:
            return "echo"  # Just echo back
        elif vocab_size < 50 or pattern_count < 10:
            return "pattern_aware"  # Acknowledge patterns
        elif vocab_size < 200:
            return "simple_generation"  # Simple pattern-based responses
        else:
            return "conversational"  # More sophisticated responses

    def generateResponse(self, shepherdInput):
        """Generate appropriate response based on current stage"""
        self.stage = self._determineStage()  # Update stage

        if self.stage == "echo":
            return self._echoResponse(shepherdInput)
        elif self.stage == "pattern_aware":
            return self._patternAwareResponse(shepherdInput)
        elif self.stage == "simple_generation":
            return self._simpleGenerativeResponse(shepherdInput)
        else:
            return self._conversationalResponse(shepherdInput)

    def _echoResponse(self, text):
        """Stage 1: Simple acknowledgment"""
        return f"Acknowledged: {text}"

    def _patternAwareResponse(self, text):
        """Stage 2: Show awareness of patterns"""
        words = self.patternEngine._tokenize(text)

        # Check for known words
        known_words = [w for w in words if w in self.patternEngine.vocabulary["words"]]
        new_words = [w for w in words if w not in self.patternEngine.vocabulary["words"]]

        if new_words and known_words:
            return f"I recognize {len(known_words)} words. Learning {len(new_words)} new ones."
        elif new_words:
            return f"New words for me: {', '.join(new_words[:3])}"
        else:
            # Check for known patterns
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i + 1]}"
                if bigram in self.patternEngine.ngrams["bigrams"]:
                    count = self.patternEngine.ngrams["bigrams"][bigram]
                    if count > 2:
                        return f"I've seen '{bigram}' {count} times now."

            return f"I understand: {text}"

    def _simpleGenerativeResponse(self, text):
        """Stage 3: Simple pattern-based generation"""
        words = self.patternEngine._tokenize(text)

        # Look for patterns in recent context
        context = self.patternEngine.getRecentContext(5)

        # Try to find similar past exchanges
        for ctx in reversed(context):
            ctx_words = ctx["words"]
            overlap = set(words) & set(ctx_words)

            if len(overlap) > 0:
                # Found contextual similarity
                common = list(overlap)[:2]
                return f"This relates to {' and '.join(common)}."

        # Check for repeated phrases
        most_common = self.patternEngine.getMostCommonPatterns(5)
        if most_common["bigrams"]:
            pattern, count = most_common["bigrams"][0]
            if count > 5:
                return f"Pattern forming: '{pattern}' appears often."

        return f"Processing: {text}"

    def _conversationalResponse(self, text):
        """Stage 4: More sophisticated responses"""
        words = self.patternEngine._tokenize(text)

        # More advanced pattern matching and response generation
        # This is where you'd implement more sophisticated NLP

        # For now, acknowledge with context
        vocab_size = self.patternEngine.vocabulary["unique_words"]
        total_words = self.patternEngine.vocabulary["total_words"]

        return f"Understanding deepens. {vocab_size} words learned from {total_words} total."


class ConfigManager:
    """Manages system configuration and identity"""

    def __init__(self):
        self.configPath = Path("system_config.json")
        self.config = self._load()

    def _load(self):
        """Load configuration from file"""
        if self.configPath.exists():
            try:
                with open(self.configPath, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass

        return {
            "initialized": False,
            "shepherd_name": None,
            "protege_name": None,
            "created_date": None
        }

    def save(self):
        """Save configuration to file"""
        with open(self.configPath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

    def isInitialized(self):
        """Check if system has been initialized"""
        return self.config.get("initialized", False)

    def initialize(self, shepherdName, protegeName):
        """Initialize system with names"""
        self.config["initialized"] = True
        self.config["shepherd_name"] = shepherdName
        self.config["protege_name"] = protegeName
        self.config["created_date"] = datetime.now().isoformat()
        self.save()

    def getShepherdName(self):
        return self.config.get("shepherd_name")

    def getProtegeName(self):
        return self.config.get("protege_name")


class JournalManager:
    """Manages daily markdown journal files"""

    def __init__(self, shepherdName, protegeName):
        self.shepherdName = shepherdName
        self.protegeName = protegeName
        self.journalDir = Path("journals")
        self.journalDir.mkdir(exist_ok=True)

        self.currentDate = None
        self.currentJournalPath = None
        self.sessionStartTime = datetime.now()

        self._ensureTodaysJournal()

    def _ensureTodaysJournal(self):
        """Create or open today's journal file"""
        today = datetime.now().strftime("%Y-%m-%d")

        if self.currentDate != today:
            self.currentDate = today
            self.currentJournalPath = self.journalDir / f"journal_{today}.md"

            if not self.currentJournalPath.exists():
                self._createJournalHeader()

    def _createJournalHeader(self):
        """Create journal file with header"""
        header = f"""# Journal - {datetime.now().strftime("%B %d, %Y")}
# Shepherd: {self.shepherdName}
# Constructed Protégé: {self.protegeName}

---

"""
        with open(self.currentJournalPath, 'w', encoding='utf-8') as f:
            f.write(header)

    def writeEntry(self, mode, content, typingStats=None):
        """Write an entry to the journal"""
        self._ensureTodaysJournal()

        timestamp = datetime.now().strftime("%H:%M")

        entry = f"**{timestamp}** - "

        if mode == "WATCH":
            entry += f"Watching shepherd. {content}"
        elif mode == "PROMPT":
            entry += f"{content}"
        elif mode == "INTERNAL":
            entry += f"*{content}*"
        elif mode == "MILESTONE":
            entry += f"🎯 **MILESTONE**: {content}"

        if typingStats:
            stats = typingStats.getStats()
            if stats:
                state = typingStats.getEmotionalState()
                entry += f" Typing: {stats['avgInterval']:.0f}ms, {state}."

        entry += "\n\n"

        with open(self.currentJournalPath, 'a', encoding='utf-8') as f:
            f.write(entry)

    def writeSessionSummary(self, observations, patternEngine):
        """Write summary section at end of session"""
        self._ensureTodaysJournal()

        summary = f"""---

## Session Summary ({self.sessionStartTime.strftime("%H:%M")} - {datetime.now().strftime("%H:%M")})

### Statistics
- Vocabulary: {patternEngine.vocabulary["unique_words"]} unique words
- Total words processed: {patternEngine.vocabulary["total_words"]}
- Patterns recognized: {len(patternEngine.ngrams["bigrams"])} bigrams, {len(patternEngine.ngrams["trigrams"])} trigrams

### Most Common Words
"""
        for word, count in patternEngine.getMostCommonWords(10):
            summary += f"- {word}: {count} times\n"

        summary += "\n### Session Observations\n"
        for observation in observations:
            summary += f"- {observation}\n"

        summary += "\n"

        with open(self.currentJournalPath, 'a', encoding='utf-8') as f:
            f.write(summary)


class TrackedLineEdit(QLineEdit):
    """QLineEdit that tracks typing signatures"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent

    def keyPressEvent(self, event):
        """Intercept key presses for signature tracking"""
        if event.isAutoRepeat():
            super().keyPressEvent(event)
            return

        key = event.key()
        timestamp = datetime.now()

        if key == Qt.Key.Key_Tab:
            self.parent.toggleInputMode()
            event.accept()
            return

        if key == Qt.Key.Key_Escape:
            self.parent.close()
            event.accept()
            return

        if self.parent.inputActive:
            self.parent._trackTypingSignature(key, timestamp)
        else:
            self.parent._observeKeyPress(event, timestamp)

        if self.parent.inputActive:
            super().keyPressEvent(event)
        else:
            event.accept()

    def keyReleaseEvent(self, event):
        """Track key releases"""
        if event.isAutoRepeat():
            super().keyReleaseEvent(event)
            return

        key = event.key()
        timestamp = datetime.now()

        if self.parent.inputActive:
            self.parent.typingSignature.recordKeyRelease(key, timestamp)

        super().keyReleaseEvent(event)


class System(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowTitle("The System")
        self.resize(860, 560)

        self.dragPosition = None
        self.resizing = False
        self.resizeStartPos = None
        self.resizeStartGeometry = None

        self.inputActive = True

        self.typingSignature = TypingSignature()

        # Configuration manager
        self.configManager = ConfigManager()

        # Bootstrap state
        self.isBootstrapping = not self.configManager.isInitialized()
        self.bootstrapStep = 0
        self.shepherdName = self.configManager.getShepherdName()
        self.protegeName = self.configManager.getProtegeName()
        self.journalManager = None
        self.patternEngine = None
        self.responseEngine = None

        # Observation tracking
        self.lastInteractionTime = datetime.now()
        self.sessionObservations = []

        # Stats display timer
        self.statsTimer = QTimer(self)
        self.statsTimer.timeout.connect(self._periodicJournalUpdate)
        self.statsTimer.start(30000)  # Every 30 seconds

        # ---------- Root layout ----------
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Header
        if self.isBootstrapping:
            self.title = QLabel("◼ System")
        else:
            self.title = QLabel(f"◼ {self.protegeName}")
        self.title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        self.title.setStyleSheet("color: #a78bfa;")
        root.addWidget(self.title)

        # ---------- INPUT ----------
        self.input = TrackedLineEdit(self)
        if self.isBootstrapping:
            self.input.setPlaceholderText("awaiting initialization...")
        else:
            self.input.setPlaceholderText("type something...")
        self.input.setFont(QFont("Arial", 11))
        self.input.returnPressed.connect(self._sendClicked)
        self.input.setStyleSheet("""
            QLineEdit {
                background: rgba(11, 15, 20, 0.45);
                color: #e6e6e6;
                border: 0px;
                border-radius: 10px;
                padding: 10px 12px;
                font-size: 16px;
            }
            QLineEdit::placeholder {
                color: rgba(230, 230, 230, 0.35);
            }
        """)
        root.addWidget(self.input)

        # ---------- DISPLAY ----------
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        self.display.setFont(QFont("Consolas", 11))
        self.display.setFrameStyle(QTextEdit.Shape.NoFrame)
        self.display.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.display.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.display.setStyleSheet("""
            QTextEdit {
                background: transparent;
                color: #e6e6e6;
                border: 0px;
                border-radius: 10px;
                padding: 6px 4px;
            }
        """)
        root.addWidget(self.display, 1)

        self.setStyleSheet("System { background: transparent; }")

        # Initialize or resume
        if self.isBootstrapping:
            self.append("sys", "System requires initialization.")
            self.append("sys", "What is your name, shepherd?")
        else:
            # Already initialized - load systems
            self.patternEngine = PatternRecognitionEngine(f"training_data/{self.shepherdName}_{self.protegeName}")
            self.responseEngine = ResponseEngine(self.patternEngine, self.protegeName)
            self.journalManager = JournalManager(self.shepherdName, self.protegeName)
            self.journalManager.writeEntry("INTERNAL", "Session resumed.")
            self.append("sys", f"Welcome back, {self.shepherdName}.")

            # Show learning progress
            vocab_size = self.patternEngine.vocabulary["unique_words"]
            if vocab_size > 0:
                self.append("sys", f"Current vocabulary: {vocab_size} words learned.")

    # -----------------------------
    # Bootstrap Process
    # -----------------------------
    def _handleBootstrap(self, text):
        """Handle bootstrap initialization"""
        if self.bootstrapStep == 0:
            self.shepherdName = text
            self.append("sys", f"Welcome, {self.shepherdName}.")
            self.append("sys", "What will you name your Constructed Protégé?")
            self.bootstrapStep = 1

        elif self.bootstrapStep == 1:
            self.protegeName = text
            self.append("sys", f"{self.protegeName} initialized.")

            # Save configuration
            self.configManager.initialize(self.shepherdName, self.protegeName)

            # Initialize learning systems
            self.patternEngine = PatternRecognitionEngine(f"training_data/{self.shepherdName}_{self.protegeName}")
            self.responseEngine = ResponseEngine(self.patternEngine, self.protegeName)
            self.journalManager = JournalManager(self.shepherdName, self.protegeName)

            self.journalManager.writeEntry("INTERNAL",
                                           f"Constructed Protégé initialized. Shepherd identified as {self.shepherdName}. Beginning observation.")

            self.isBootstrapping = False
            self.title.setText(f"◼ {self.protegeName}")
            self.input.setPlaceholderText("type something...")

            self.append("sys", "Initialization complete. Learning systems active.")
            self.append("sys", "I will learn from every interaction.")

    # -----------------------------
    # Input Mode Toggle
    # -----------------------------
    def toggleInputMode(self):
        """Toggle between PROMPT and WATCH modes"""
        if self.isBootstrapping:
            return

        self.inputActive = not self.inputActive

        if self.inputActive:
            self.title.setText(f"◼ {self.protegeName}")
            self.title.setStyleSheet("color: #a78bfa;")
            self.input.setReadOnly(False)
            self.input.setPlaceholderText("type something...")
            self.append("sys", "Mode: PROMPT - Direct interaction")
            self.journalManager.writeEntry("INTERNAL", "Switched to PROMPT mode.")
        else:
            self.title.setText(f"◻ {self.protegeName}")
            self.title.setStyleSheet("color: #f59e0b;")
            self.input.setReadOnly(True)
            self.input.setPlaceholderText("observing...")
            self.append("sys", "Mode: WATCH - Silent observation")
            self.journalManager.writeEntry("INTERNAL", "Switched to WATCH mode.")

        self.typingSignature.reset()

    # -----------------------------
    # Tracking Methods
    # -----------------------------
    def _trackTypingSignature(self, key, timestamp):
        """Track typing patterns when input is active"""
        self.typingSignature.recordKeyPress(key, timestamp)
        self.lastInteractionTime = timestamp

    def _observeKeyPress(self, event, timestamp):
        """Observe keystrokes when in watch mode"""
        key = event.key()
        keyName = self._getKeyName(key)

        self.append("sys", f"[WATCH] Key observed: {keyName}")

        self.lastInteractionTime = timestamp

    def _periodicJournalUpdate(self):
        """Periodic journal updates based on activity"""
        if self.isBootstrapping or not self.journalManager:
            return

        timeSinceInteraction = (datetime.now() - self.lastInteractionTime).total_seconds()

        if timeSinceInteraction < 60:  # Active in last minute
            stats = self.typingSignature.getStats()
            if stats and stats['totalKeys'] > 5:
                mode = "PROMPT" if self.inputActive else "WATCH"

                if self.inputActive:
                    content = f"Shepherd actively typing. {stats['totalKeys']} keys this session."
                else:
                    content = f"Shepherd working elsewhere. Observing typing patterns."

                self.journalManager.writeEntry(mode, content, self.typingSignature)

    def _getKeyName(self, key):
        """Convert key code to readable name"""
        keyMap = {
            Qt.Key.Key_Space: "Space",
            Qt.Key.Key_Return: "Enter",
            Qt.Key.Key_Enter: "Enter",
            Qt.Key.Key_Backspace: "Backspace",
            Qt.Key.Key_Delete: "Delete",
            Qt.Key.Key_Tab: "Tab",
            Qt.Key.Key_Escape: "Esc",
            Qt.Key.Key_Up: "Up",
            Qt.Key.Key_Down: "Down",
            Qt.Key.Key_Left: "Left",
            Qt.Key.Key_Right: "Right",
        }

        if key in keyMap:
            return keyMap[key]

        text = chr(key) if 32 <= key <= 126 else f"Key_{key}"
        return text

    # -----------------------------
    # Move/resize behavior
    # -----------------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragPosition = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self.resizing = True
            self.resizeStartPos = event.globalPosition().toPoint()
            self.resizeStartGeometry = self.geometry()
            event.accept()

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.dragPosition:
            self.move(event.globalPosition().toPoint() - self.dragPosition)
            event.accept()
        elif (event.buttons() & Qt.MouseButton.RightButton) and self.resizing:
            delta = event.globalPosition().toPoint() - self.resizeStartPos
            newW = max(420, self.resizeStartGeometry.width() + delta.x())
            newH = max(300, self.resizeStartGeometry.height() + delta.y())
            self.resize(newW, newH)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragPosition = None
        elif event.button() == Qt.MouseButton.RightButton:
            self.resizing = False
            self.resizeStartPos = None
            self.resizeStartGeometry = None
        event.accept()

    # -----------------------------
    # Chat functions
    # -----------------------------
    def append(self, role: str, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        roleColors = {
            "user": "#22c55e",
            "bot": "#f59e0b",
            "sys": "#a78bfa",
        }
        c = roleColors.get(role, "#e6e6e6")

        html = (
            f'<span style="color: rgba(148,163,184,0.9);">[{ts}] </span>'
            f'<span style="color:{c};">{self._esc(role)} &gt; {self._esc(text)}</span>'
        )

        self.display.append(html)
        self.display.moveCursor(QTextCursor.MoveOperation.End)
        self.display.ensureCursorVisible()

    def _sendClicked(self):
        text = self.input.text().strip()
        if not text:
            return
        self.input.clear()

        if self.isBootstrapping:
            self.append("user", text)
            self._handleBootstrap(text)
        else:
            self.append(self.shepherdName, text)

            # Process input through pattern engine
            analysis = self.patternEngine.processInput(text)

            # Log to journal
            self.journalManager.writeEntry("PROMPT",
                                           f"Shepherd said: \"{text}\"",
                                           self.typingSignature)

            # Check for new milestones
            milestones = self.patternEngine.checkNewMilestones()
            for milestone in milestones:
                milestone_text = self._formatMilestone(milestone)
                self.append("sys", f"🎯 {milestone_text}")
                self.journalManager.writeEntry("MILESTONE", milestone_text)

            # Generate response using response engine
            response = self.responseEngine.generateResponse(text)
            self.append(self.protegeName, response)

            self.journalManager.writeEntry("PROMPT",
                                           f"I responded: \"{response}\"")

            # Log learning progress if new words found
            if analysis["new_words"]:
                self.journalManager.writeEntry("INTERNAL",
                                               f"Learned new words: {', '.join(analysis['new_words'][:5])}")

    def _formatMilestone(self, milestone):
        """Format milestone for display"""
        m_type = milestone["type"]
        data = milestone["data"]

        if m_type == "first_word":
            return f"First word learned: '{data['word']}'"
        elif m_type == "first_repeated_word":
            return f"First repeated word: '{data['word']}'"
        elif m_type.startswith("vocabulary_"):
            threshold = m_type.split("_")[1]
            return f"Vocabulary milestone: {threshold} unique words!"
        elif m_type == "first_pattern_recognized":
            return f"First pattern recognized: '{data['pattern']}' (seen {data['count']} times)"
        else:
            return f"Milestone achieved: {m_type}"

    def closeEvent(self, event):
        """Write session summary on close"""
        if self.journalManager and not self.isBootstrapping:
            self.sessionObservations.append(
                f"Session duration: {(datetime.now() - self.journalManager.sessionStartTime).seconds // 60} minutes"
            )
            stats = self.typingSignature.getStats()
            if stats:
                self.sessionObservations.append(
                    f"Typing signature: {stats['avgInterval']:.0f}ms average, {stats['totalKeys']} total keys"
                )

            # Add learning summary
            vocab_size = self.patternEngine.vocabulary["unique_words"]
            total_words = self.patternEngine.vocabulary["total_words"]
            self.sessionObservations.append(
                f"Learning progress: {vocab_size} vocabulary, {total_words} words processed"
            )

            self.journalManager.writeSessionSummary(self.sessionObservations, self.patternEngine)

        super().closeEvent(event)

    @staticmethod
    def _esc(s: str) -> str:
        return (s.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;"))


def main():
    app = QApplication(sys.argv)

    system = System()
    system.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()