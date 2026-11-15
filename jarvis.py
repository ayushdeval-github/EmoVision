"""
Jarvis AI assistant for EmoVision.

Architecture:
- Groq Chat Completions: fast, multi-turn everyday voice conversation.
- Gemini Interactions API: deeper response after a sustained facial-expression
  estimate. Uses previous_interaction_id to retain Gemini-side conversation state.
- Webcam/model logic remains in run_fixed.py.
- run_main.py forwards run_fixed.py model predictions to update_emotion().

Required environment variables:
    GROQ_API_KEY
    GEMINI_API_KEY

Optional:
    GROQ_MODEL=openai/gpt-oss-20b
    GEMINI_MODEL=gemini-3.5-flash-lite
    GEMINI_SUSTAINED_SECONDS=180
    EMOTION_MIN_CONFIDENCE=45
    EMOTION_STABLE_FRAMES=8
"""

from __future__ import annotations

import os
import re
import time
import queue
import threading
import webbrowser
from collections import deque
from pathlib import Path
from typing import Optional

# Load .env from the same directory as this file, independent of terminal cwd.
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().with_name(".env"), override=False)
except ImportError:
    pass

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from google import genai
except ImportError:
    genai = None

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import speech_recognition as sr
except ImportError:
    sr = None


SYSTEM_PROMPT = """
You are Jarvis, a warm, respectful, emotionally aware voice assistant.
Speak naturally, casually, and clearly. For spoken replies, prefer 1-3 concise
sentences. Keep track of the conversation history supplied to you. Adapt your
tone gently to the user's words and the current facial-expression estimate.
A facial-expression classifier is uncertain and does not establish how the
user truly feels. Never diagnose, pressure, shame, or lecture the user.
Ask thoughtful, non-leading questions when useful. Avoid repetitive clichés.
"""

GEMINI_EMPATHY_PROMPT = """
You are Jarvis, a thoughtful and compassionate voice companion.
The camera-based emotion classifier has continuously estimated the facial
expression described below. This is only an uncertain visual signal; it does
not prove the user's actual internal emotional state.

Prepare a meaningful, natural spoken response of about 3-5 sentences.
Acknowledge uncertainty gently without sounding robotic. Be empathetic, not
dramatic. Do not diagnose or assume a cause. Offer one gentle check-in question
or one optional, practical supportive step. Avoid generic clichés.
"""


class JarvisAssistant:
    """Voice assistant coordinated with live emotion estimates from run_main.py."""

    def __init__(self, report_callback=None):
        self.report_callback = report_callback

        # Provider configuration. These defaults are current provider-supported
        # replacements rather than the retired Groq model in the sample.
        self.groq_model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.gemini_model = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

        self.sustained_seconds = max(
            30, int(os.getenv("GEMINI_SUSTAINED_SECONDS", "180"))
        )
        self.face_stale_seconds = max(
            1.0, float(os.getenv("FACE_STALE_SECONDS", "2.5"))
        )
        self.min_confidence = min(
            100.0, max(0.0, float(os.getenv("EMOTION_MIN_CONFIDENCE", "45")))
        )
        self.stable_frames_required = max(
            1, int(os.getenv("EMOTION_STABLE_FRAMES", "8"))
        )
        # Minimum time between emotion announcements (seconds)
        self.emotion_announce_cooldown = max(
            5.0, float(os.getenv("EMOTION_ANNOUNCE_COOLDOWN", "15.0"))
        )

        self.current_emotion = "Not detected"
        self.current_confidence = 0.0

        self._state_lock = threading.RLock()
        self._groq_lock = threading.Lock()
        self._gemini_lock = threading.Lock()
        self._history = deque(maxlen=20)

        # Stable emotion tracking to prevent speaking on every noisy prediction.
        self._candidate_emotion = None
        self._candidate_frames = 0
        self._last_announced_emotion = None
        self._last_emotion_announce_time = 0.0
        self._last_face_update = 0.0
        self._emotion_started_at = None
        self._last_emotion_for_episode = None
        self._deep_response_sent = False
        self._deep_request_running = False
        self._recent_emotions = deque(maxlen=20)

        # Gemini server-side state ID, set after first successful Interactions call.
        self._gemini_previous_interaction_id: Optional[str] = None

        self._stop_event = threading.Event()
        self._speech_queue = queue.Queue()
        self._speaking_event = threading.Event()
        self._listener_thread = None
        self._speech_thread = None
        self._monitor_thread = None

        # Initialize Groq client.
        groq_key = os.getenv("GROQ_API_KEY")
        self._groq = None
        if Groq is None:
            print("[Jarvis] Missing package. Install it with: pip install groq")
        elif not groq_key:
            print("[Jarvis] GROQ_API_KEY is missing. Groq conversation is disabled.")
        else:
            try:
                self._groq = Groq(api_key=groq_key)
                print(f"[Jarvis] Groq client ready. Model: {self.groq_model}")
            except Exception as exc:
                print(f"[Jarvis] Groq initialization failed: {exc}")

        # Initialize Google Gen AI client.
        gemini_key = os.getenv("GEMINI_API_KEY")
        self._gemini = None
        if genai is None:
            print("[Jarvis] Missing package. Install it with: pip install --upgrade google-genai")
        elif not gemini_key:
            print("[Jarvis] GEMINI_API_KEY is missing. Gemini reflections are disabled.")
        else:
            try:
                self._gemini = genai.Client(api_key=gemini_key)
                print(f"[Jarvis] Gemini client ready. Model: {self.gemini_model}")
            except Exception as exc:
                print(f"[Jarvis] Gemini initialization failed: {exc}")

        # Start independent speech and emotion-monitor workers.
        self._speech_thread = threading.Thread(
            target=self._speech_worker, name="JarvisSpeech", daemon=True
        )
        self._monitor_thread = threading.Thread(
            target=self._emotion_monitor, name="JarvisEmotionMonitor", daemon=True
        )
        self._speech_thread.start()
        self._monitor_thread.start()

        print("[Jarvis] Assistant initialized.", flush=True)

    # ------------------------------------------------------------------
    # Speech output: every assistant response appears in the terminal
    # and is also sent to pyttsx3 for audible playback.
    # ------------------------------------------------------------------
    def speak(self, message: str, wait=False) -> None:
        """
        Queue a message for speech output.
        
        Args:
            message: Text to speak
            wait: If True, blocks until this specific message finishes speaking
        """
        if not message:
            return
        message = str(message).strip()
        if not message:
            return
        print(f"\n[Jarvis] {message}\n", flush=True)
        
        # Create a completion event for this specific message if waiting is requested
        done_event = threading.Event() if wait else None
        self._speech_queue.put((message, done_event))
        
        if wait and done_event:
            # Wait up to 30 seconds for this message to complete
            done_event.wait(timeout=30.0)

    def _speech_worker(self) -> None:
        if pyttsx3 is None:
            print("[Jarvis] pyttsx3 is not installed; voice output is unavailable.")
            return

        while not self._stop_event.is_set() or not self._speech_queue.empty():
            try:
                item = self._speech_queue.get(timeout=0.2)
                message, done_event = item if isinstance(item, tuple) else (item, None)
            except queue.Empty:
                # Clear speaking flag when queue is truly empty
                if self._speech_queue.empty():
                    self._speaking_event.clear()
                continue

            try:
                self._speaking_event.set()
                
                # FIX: Reinitialize engine for each message (ensures proper completion)
                engine = pyttsx3.init()
                engine.setProperty("rate", 170)
                voices = engine.getProperty("voices")
                if voices:
                    engine.setProperty("voice", voices[0].id)
                
                # Speak the message
                engine.say(message)
                engine.runAndWait()
                
                # Cleanup engine
                del engine
                
                # Small delay for stability
                time.sleep(0.2)
                
            except Exception as exc:
                print(f"[Jarvis] Voice playback error: {exc}")
            finally:
                self._speech_queue.task_done()
                # Signal completion for this specific message
                if done_event:
                    done_event.set()
                # Clear speaking flag only when queue becomes empty
                if self._speech_queue.empty():
                    self._speaking_event.clear()

    def wait_for_speech_completion(self, timeout=10.0):
        """Wait for all queued speech to complete."""
        deadline = time.monotonic() + timeout
        while self._speaking_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)

    # ------------------------------------------------------------------
    # Live emotion feed, called by run_main.py for each prediction made
    # by the model object loaded and used by run_fixed.py.
    # ------------------------------------------------------------------
    def update_emotion(self, emotion: str, confidence: float) -> None:
        emotion = str(emotion).strip().title()
        confidence = float(confidence)
        now = time.monotonic()

        with self._state_lock:
            self._last_face_update = now

            if emotion in {"Unknown", "Not Detected", "Uncertain"} or confidence < self.min_confidence:
                return

            if emotion == self._candidate_emotion:
                self._candidate_frames += 1
            else:
                self._candidate_emotion = emotion
                self._candidate_frames = 1

            if self._candidate_frames < self.stable_frames_required:
                return

            self.current_emotion = emotion
            self.current_confidence = confidence

            if emotion != self._last_emotion_for_episode:
                self._last_emotion_for_episode = emotion
                self._emotion_started_at = now
                self._deep_response_sent = False
                self._recent_emotions.append(emotion)

            # Cooldown-based announcement to prevent spam
            time_since_last_announce = now - self._last_emotion_announce_time
            should_announce = (
                emotion != self._last_announced_emotion 
                and time_since_last_announce >= self.emotion_announce_cooldown
            )
            
            if should_announce:
                self._last_announced_emotion = emotion
                self._last_emotion_announce_time = now

        if should_announce:
            print(
                f"[Emotion Detection] {emotion} "
                f"({confidence:.1f}% confidence)",
                flush=True,
            )
            self.speak(
                f"I detect a {emotion.lower()} expression with {confidence:.0f} percent confidence."
            )

    def _emotion_monitor(self) -> None:
        """Trigger Gemini once per continuous, stable non-neutral expression."""
        while not self._stop_event.wait(0.5):
            now = time.monotonic()
            deep_request = None

            with self._state_lock:
                # Do not count time when the camera has stopped reporting a face.
                if (
                    self._last_face_update
                    and now - self._last_face_update > self.face_stale_seconds
                ):
                    self.current_emotion = "Not detected"
                    self.current_confidence = 0.0
                    self._candidate_emotion = None
                    self._candidate_frames = 0
                    self._last_announced_emotion = None
                    self._last_emotion_for_episode = None
                    self._emotion_started_at = None
                    self._deep_response_sent = False
                    self._last_face_update = 0.0

                emotion = self.current_emotion
                started = self._emotion_started_at

                if (
                    emotion not in {"Not detected", "Neutral", "Unknown"}
                    and started is not None
                    and now - started >= self.sustained_seconds
                    and not self._deep_response_sent
                    and not self._deep_request_running
                ):
                    self._deep_response_sent = True
                    self._deep_request_running = True
                    deep_request = (
                        emotion,
                        now - started,
                        self.current_confidence,
                        list(self._recent_emotions),
                    )

            if deep_request:
                threading.Thread(
                    target=self._gemini_sustained_emotion_response,
                    args=deep_request,
                    name="JarvisGeminiReflection",
                    daemon=True,
                ).start()

    # ------------------------------------------------------------------
    # Groq: fast conversational turns using Chat Completions.
    # ------------------------------------------------------------------
    def chat(self, user_text: str) -> str:
        if self._groq is None:
            return (
                "My Groq conversation service is not configured. "
                "Please check GROQ_API_KEY and the Groq package."
            )

        with self._state_lock:
            emotion = self.current_emotion
            confidence = self.current_confidence

        system_message = (
            SYSTEM_PROMPT
            + f"\nCurrent facial-expression estimate: {emotion} "
            + f"with {confidence:.0f}% classifier confidence. "
            + "Treat this as an uncertain visual cue, not a fact."
        )

        with self._groq_lock:
            messages = [{"role": "system", "content": system_message}]
            messages.extend(list(self._history))
            messages.append({"role": "user", "content": user_text})

            try:
                completion = self._groq.chat.completions.create(
                    model=self.groq_model,
                    messages=messages,
                    temperature=0.6,
                    max_completion_tokens=350,
                    timeout=25.0,
                )
                answer = (
                    completion.choices[0].message.content or ""
                ).strip()

                if not answer:
                    answer = "I'm listening. Could you tell me a little more?"

                self._history.append({"role": "user", "content": user_text})
                self._history.append({"role": "assistant", "content": answer})
                return answer

            except Exception as exc:
                print(f"[Jarvis] Groq API request failed: {exc}", flush=True)
                return (
                    "I'm having trouble connecting to my conversation service. "
                    "Please check the Groq API key and internet connection."
                )

    # ------------------------------------------------------------------
    # Gemini: stateful Interactions API for sustained emotional reflections.
    # ------------------------------------------------------------------
    def _gemini_sustained_emotion_response(
        self,
        emotion: str,
        duration_seconds: float,
        confidence: float,
        recent_emotions: list,
    ) -> None:
        try:
            if self._gemini is None:
                print(
                    "[Jarvis] Gemini is not configured. "
                    "Cannot create the sustained-emotion reflection.",
                    flush=True,
                )
                return

            prompt = (
                GEMINI_EMPATHY_PROMPT
                + "\n\n"
                + f"Current sustained expression estimate: {emotion}.\n"
                + f"Estimated duration: {int(duration_seconds)} seconds.\n"
                + f"Latest classifier confidence: {confidence:.1f}%.\n"
                + "Recent stable expression labels: "
                + (", ".join(recent_emotions) if recent_emotions else "not available")
                + "\n\nWrite the spoken response now."
            )

            # This follows the user's supplied Interactions API pattern:
            # create the first Interaction, then chain later requests by ID.
            with self._gemini_lock:
                create_args = {
                    "model": self.gemini_model,
                    "input": prompt,
                }
                if self._gemini_previous_interaction_id:
                    create_args["previous_interaction_id"] = (
                        self._gemini_previous_interaction_id
                    )

                interaction = self._gemini.interactions.create(**create_args)
                self._gemini_previous_interaction_id = interaction.id
                response_text = (interaction.output_text or "").strip()

            if response_text:
                self.speak(response_text)
            else:
                print("[Jarvis] Gemini returned an empty response.", flush=True)

        except Exception as exc:
            print(f"[Jarvis] Gemini Interactions API request failed: {exc}", flush=True)
        finally:
            with self._state_lock:
                self._deep_request_running = False

    # ------------------------------------------------------------------
    # Voice commands and microphone listener.
    # ------------------------------------------------------------------
    def _handle_command(self, command: str) -> None:
        command = command.strip()
        lowered = command.lower()

        if not lowered:
            self.speak("I did not catch that. Please try again.")
            return

        if "what emotion" in lowered or "my emotion" in lowered or "what do you see" in lowered:
            with self._state_lock:
                emotion = self.current_emotion
                confidence = self.current_confidence
            self.speak(
                f"The live camera detection currently shows {emotion}, "
                f"with {confidence:.0f} percent confidence."
            )
        elif "how are you" in lowered:
            self.speak("I'm working properly and ready to help you.")

        elif "open youtube" in lowered:
            self.speak("Opening YouTube now.", wait=True)
            try:
                webbrowser.open("https://www.youtube.com")
                time.sleep(0.5)  # Brief delay to ensure browser opens
                self.speak("YouTube opened successfully.")
            except Exception as exc:
                print(f"[Jarvis] Failed to open YouTube: {exc}")
                self.speak("I encountered an error opening YouTube.")

        elif "open google" in lowered:
            self.speak("Opening Google now.", wait=True)
            try:
                webbrowser.open("https://www.google.com")
                time.sleep(0.5)
                self.speak("Google opened successfully.")
            except Exception as exc:
                print(f"[Jarvis] Failed to open Google: {exc}")
                self.speak("I encountered an error opening Google.")

        elif "make pdf" in lowered or "create report" in lowered:
            if self.report_callback:
                self.speak("Creating your emotion report now.", wait=True)
                try:
                    self.report_callback()
                    self.speak("Report created successfully.")
                except Exception as exc:
                    print(f"[Jarvis] Report callback failed: {exc}", flush=True)
                    self.speak("I could not create the report. An error occurred.")
            else:
                self.speak("The report generator is not connected.")

        elif "open whatsapp" in lowered:
            self.speak("Opening WhatsApp now.", wait=True)
            try:
                webbrowser.open("https://web.whatsapp.com")
                time.sleep(0.5)
                self.speak("WhatsApp opened successfully.")
            except Exception as exc:
                print(f"[Jarvis] Failed to open WhatsApp: {exc}")
                self.speak("I encountered an error opening WhatsApp.")

        elif "open calculator" in lowered:
            self.speak("Opening calculator.", wait=True)
            try:
                import subprocess
                subprocess.Popen("calc.exe")
                time.sleep(0.5)
                self.speak("Calculator opened.")
            except Exception as exc:
                print(f"[Jarvis] Failed to open calculator: {exc}")
                self.speak("I could not open the calculator.")

        elif "open notepad" in lowered:
            self.speak("Opening Notepad.", wait=True)
            try:
                import subprocess
                subprocess.Popen("notepad.exe")
                time.sleep(0.5)
                self.speak("Notepad opened.")
            except Exception as exc:
                print(f"[Jarvis] Failed: {exc}")
                self.speak("Could not open Notepad.")

        elif "search" in lowered and ("google" in lowered or "on google" in lowered):
            # Extract the search query
            import urllib.parse
            query = None
            
            # Pattern 1: "search [query] on google"
            if "on google" in lowered:
                query = lowered.split("search", 1)[1].split("on google")[0].strip()
            # Pattern 2: "search google for [query]"
            elif "google for" in lowered:
                query = lowered.split("google for", 1)[1].strip()
            # Pattern 3: "search for [query]"
            elif "search for" in lowered:
                query = lowered.split("search for", 1)[1].replace("google", "").strip()
            # Pattern 4: "google search [query]"
            elif "google search" in lowered:
                query = lowered.split("google search", 1)[1].strip()
            # Pattern 5: Just "search [query]" (if google mentioned)
            else:
                query = lowered.replace("search", "").replace("google", "").replace("on", "").strip()
            
            if query and len(query) > 1:
                self.speak(f"Searching Google for {query}.", wait=True)
                try:
                    search_url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                    webbrowser.open(search_url)
                    time.sleep(0.5)
                    self.speak("Search results opened.")
                except Exception as exc:
                    print(f"[Jarvis] Search failed: {exc}")
                    self.speak("Search failed.")
            else:
                self.speak("What would you like me to search for?")

        elif "play" in lowered and ("youtube" in lowered or "on youtube" in lowered):
            # Extract song name - everything after "play" and before "on youtube" or "youtube"
            import re
            import urllib.parse
            
            # Try to extract song name
            song = None
            
            # Pattern 1: "play [song] on youtube"
            match = re.search(r'play\s+(.+?)\s+on\s+youtube', lowered)
            if match:
                song = match.group(1).strip()
            
            # Pattern 2: "play [song] youtube"
            if not song:
                match = re.search(r'play\s+(.+?)\s+youtube', lowered)
                if match:
                    song = match.group(1).strip()
            
            # Pattern 3: Just "play [song]" (if youtube mentioned somewhere)
            if not song and "play" in lowered:
                parts = lowered.split("play", 1)
                if len(parts) > 1:
                    rest = parts[1].replace("youtube", "").replace("on", "").strip()
                    if rest:
                        song = rest
            
            if song and len(song) > 1:
                self.speak(f"Playing {song} on YouTube.", wait=True)
                try:
                    search_url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(song)}"
                    webbrowser.open(search_url)
                    time.sleep(0.5)
                    self.speak("YouTube search opened.")
                except Exception as exc:
                    print(f"[Jarvis] Failed to search YouTube: {exc}")
                    self.speak("Could not open YouTube.")
            else:
                self.speak("What song would you like me to play on YouTube?")

        elif "open file explorer" in lowered or "open explorer" in lowered:
            self.speak("Opening File Explorer.", wait=True)
            try:
                import subprocess
                subprocess.Popen("explorer.exe")
                time.sleep(0.5)
                self.speak("File Explorer opened.")
            except Exception as exc:
                print(f"[Jarvis] Failed: {exc}")
                self.speak("Could not open File Explorer.")

        elif "what time" in lowered or "current time" in lowered:
            import datetime
            now = datetime.datetime.now()
            time_str = now.strftime("%I:%M %p")
            self.speak(f"The current time is {time_str}.")

        elif "what date" in lowered or "today's date" in lowered:
            import datetime
            now = datetime.datetime.now()
            date_str = now.strftime("%B %d, %Y")
            self.speak(f"Today is {date_str}.")

        elif "take screenshot" in lowered or "screenshot" in lowered:
            self.speak("Taking screenshot.", wait=True)
            try:
                import pyautogui
                import datetime
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
                pyautogui.screenshot(filename)
                self.speak(f"Screenshot saved as {filename}.")
            except Exception as exc:
                print(f"[Jarvis] Screenshot failed: {exc}")
                self.speak("Could not take screenshot.")


        elif lowered in {"stop jarvis", "stop assistant", "exit jarvis", "shutdown"}:
            self.speak("Jarvis voice assistant is shutting down now. Goodbye.")
            self.wait_for_speech_completion()
            self._stop_event.set()
        else:
            # All free-form voice conversation goes to Groq.
            response = self.chat(command)
            self.speak(response)

    @staticmethod
    def _extract_command_after_wake_word(heard: str) -> str:
        """Accept both 'Jarvis' alone and 'Jarvis, open YouTube' in one utterance."""
        match = re.search(r"\bjarvis\b[\s,!.:;-]*(.*)$", heard, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""

    def start_listener(self) -> None:
        if sr is None:
            print(
                "[Jarvis] SpeechRecognition is missing. "
                "Install SpeechRecognition and PyAudio for microphone commands.",
                flush=True,
            )
            return

        if self._listener_thread and self._listener_thread.is_alive():
            return

        self._listener_thread = threading.Thread(
            target=self._listen_loop,
            name="JarvisVoiceListener",
            daemon=True,
        )
        self._listener_thread.start()

    def _listen_loop(self) -> None:
        recognizer = sr.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8

        self.speak("Jarvis is ready. Say Jarvis, followed by your command.", wait=True)

        while not self._stop_event.is_set():
            if self._speaking_event.is_set():
                time.sleep(0.1)
                continue

            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(
                        source, timeout=2, phrase_time_limit=6
                    )

                heard = recognizer.recognize_google(audio).strip()
                if not heard:
                    continue

                print(f"[You] {heard}", flush=True)
                if "jarvis" not in heard.lower():
                    continue

                command = self._extract_command_after_wake_word(heard)

                if not command:
                    self.speak("Yes, I'm listening.", wait=True)
                    # Wait until the acknowledgement has played before opening the microphone.
                    self.wait_for_speech_completion(timeout=8)

                    with sr.Microphone() as source:
                        command_audio = recognizer.listen(
                            source, timeout=7, phrase_time_limit=12
                        )
                    command = recognizer.recognize_google(command_audio).strip()
                    print(f"[You] {command}", flush=True)

                self._handle_command(command)

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                # Silently ignore - too noisy to print every time
                continue
            except sr.RequestError as exc:
                print(f"[Jarvis] Speech recognition service error: {exc}", flush=True)
                time.sleep(2)
            except OSError as exc:
                print(f"[Jarvis] Microphone error: {exc}", flush=True)
                time.sleep(3)
            except Exception as exc:
                print(f"[Jarvis] Listener error: {exc}", flush=True)
                time.sleep(1)

    def shutdown(self) -> None:
        self._stop_event.set()

        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=1.5)

        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)

        if self._speech_thread and self._speech_thread.is_alive():
            self._speech_thread.join(timeout=3.0)

        print("[Jarvis] Shutdown complete.", flush=True)
