
"""
Main launcher for EmoVision + Jarvis.

run_fixed.py:
    Loads the emotion model, opens webcam, detects faces
    and predicts facial expressions.

jarvis.py:
    Handles Groq, Gemini, voice recognition and voice feedback.

This file connects both components.
"""

import datetime
import importlib.util
from pathlib import Path

import jarvis


# =====================================================
# PROJECT CONFIGURATION
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

RUN_FIXED_PATH = BASE_DIR / "run_fixed.py"

if not RUN_FIXED_PATH.is_file():
    raise FileNotFoundError(
        "run_fixed.py was not found. Keep run_main.py, "
        "jarvis.py and run_fixed.py in the same folder."
    )


# =====================================================
# IMPORT ORIGINAL EMOTION DETECTION FILE
# =====================================================

SPEC = importlib.util.spec_from_file_location(
    "emotion_detector",
    RUN_FIXED_PATH
)

if SPEC is None or SPEC.loader is None:
    raise ImportError(
        f"Could not load run_fixed.py: {RUN_FIXED_PATH}"
    )

detector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(detector)


# =====================================================
# GLOBAL VARIABLES
# =====================================================

emotion_history = []
assistant = None


# =====================================================
# PDF REPORT GENERATOR
# =====================================================

def create_pdf_report():

    filename = BASE_DIR / (
        f"EmoVision_Report_"
        f"{datetime.datetime.now():%Y%m%d_%H%M%S}.pdf"
    )

    try:
        from reportlab.pdfgen import canvas

        pdf = canvas.Canvas(str(filename))

        pdf.setFont("Helvetica-Bold", 16)
        pdf.drawString(
            50, 800,
            "EmoVision - Emotion Session Report"
        )

        pdf.setFont("Helvetica", 10)

        pdf.drawString(
            50, 780,
            f"Generated: {datetime.datetime.now():%d %B %Y %I:%M %p}"
        )

        pdf.drawString(
            50, 755,
            "Time                 Emotion          Confidence"
        )

        y = 735

        for timestamp, emotion, confidence in emotion_history[-40:]:

            pdf.drawString(
                50,
                y,
                f"{timestamp:<21} {emotion:<16} {confidence:.1f}%"
            )

            y -= 16

            if y < 60:
                pdf.showPage()
                pdf.setFont("Helvetica", 10)
                y = 800

        pdf.save()

        print(
            f"[EmoVision] PDF report saved: {filename}",
            flush=True
        )

        if assistant is not None:
            assistant.speak(
                f"Your emotion report was saved as {filename.name}."
            )

    except Exception as exc:

        print(
            f"[EmoVision] Could not create PDF report: {exc}",
            flush=True
        )

        if assistant is not None:
            assistant.speak(
                "I could not create the PDF report."
            )


# =====================================================
# CONNECT LIVE EMOTION PREDICTIONS TO JARVIS
# =====================================================

def attach_live_emotion_feed(model):

    """
    Connect the original model.predict() method
    to the Jarvis AI assistant.

    Original predictions remain unchanged.
    """

    import numpy as np

    original_predict = model.predict

    def predict_with_jarvis_feedback(*args, **kwargs):

        # Execute original emotion prediction.
        predictions = original_predict(*args, **kwargs)

        try:

            values = np.asarray(predictions)

            if values.ndim == 2 and values.shape[0] > 0:

                probabilities = values[0]

            elif values.ndim == 1:

                probabilities = values

            else:

                return predictions

            if probabilities.size != detector.NUM_CLASSES:

                return predictions

            # Get emotion class.
            class_index = int(np.argmax(probabilities))

            confidence = (
                float(probabilities[class_index]) * 100.0
            )

            emotion = detector.EMOTION_LABELS.get(
                class_index,
                "Unknown"
            )

            # Send live emotion to Jarvis.
            if assistant is not None:

                assistant.update_emotion(
                    emotion,
                    confidence
                )

            # Save emotion transitions for PDF reporting.
            if (
                not emotion_history
                or emotion_history[-1][1] != emotion
            ):

                emotion_history.append(
                    (
                        datetime.datetime.now().strftime("%H:%M:%S"),
                        emotion,
                        confidence
                    )
                )

        except Exception as exc:

            print(
                f"[Integration] Could not forward prediction: {exc}",
                flush=True
            )

        # Return original prediction to run_fixed.py.
        return predictions

    model.predict = predict_with_jarvis_feedback


# =====================================================
# MAIN APPLICATION
# =====================================================

def main():

    global assistant

    print("\n" + "=" * 55)
    print("       EMOVISION PRO + JARVIS AI ASSISTANT")
    print("=" * 55)

    try:

        # Initialize Jarvis AI.
        print("\n[Main] Initializing Jarvis AI...")

        assistant = jarvis.JarvisAssistant(
            report_callback=create_pdf_report
        )

        # Start voice recognition.
        assistant.start_listener()

        # Load original emotion detection model.
        print("\n[Main] Loading emotion detection model...")

        model = detector.load_or_train_model()

        print("[Main] Emotion model loaded successfully.")

        # Connect emotion model with Jarvis.
        attach_live_emotion_feed(model)

        print(
            "[Main] Jarvis connected to live emotion predictions."
        )

        print("\n" + "=" * 55)
        print("WEBCAM AND EMOTION DETECTION STARTING")
        print("Say Jarvis followed by your voice command.")
        print("Press Q in webcam window to exit.")
        print("=" * 55 + "\n")

        # Run original webcam function.
        # Webcam logic stays inside run_fixed.py.
        detector.run_webcam(model)

    except KeyboardInterrupt:

        print("\n[Main] Application stopped by user.")

    except Exception as exc:

        print(
            f"\n[Main] Application error: {exc}",
            flush=True
        )

        raise

    finally:

        if assistant is not None:
            assistant.shutdown()

        print(
            "\n[Main] EmoVision + Jarvis session ended."
        )


# =====================================================
# APPLICATION ENTRY POINT
# =====================================================

if __name__ == "__main__":

    main()
