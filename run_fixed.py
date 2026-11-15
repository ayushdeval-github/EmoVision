import os
import sys
import time
import numpy as np
import cv2


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Dense,
    Dropout,
    Flatten,
)


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ---------------- Configuration ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataa")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALID_DIR = os.path.join(DATA_DIR, "test")


BATCH_SIZE = 64
TARGET_SIZE = (48, 48)
NUM_CLASSES = 7
EPOCHS = 50
MODEL_FILENAME = os.path.join(
    BASE_DIR,
    "emotion_detection_model.keras"
)
EXPECTED_CLASSES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

# Keras flow_from_directory assigns indices in alphabetical order.
EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Sad",
    6: "Surprise",
}

# ---------------- Performance Optimization Configuration ----------------
# Frame processing optimization: Skip frames to reduce computational load
# Process every (FRAME_SKIP + 1)th frame (e.g., FRAME_SKIP=2 means process frames 0, 3, 6, 9...)
# Acceptable range: 1-5 (higher = more skipping, lower CPU usage but less responsive)
FRAME_SKIP = 2

# Frame rate control: Target frames per second for consistent video display
# Controls the display refresh rate to maintain smooth real-time performance
# Acceptable range: 15-30 FPS (lower = less CPU usage, higher = smoother video)
TARGET_FPS = 20

# Confidence threshold: Minimum confidence level to display emotion predictions
# Filters out low-confidence predictions to improve accuracy (0.0 = 0%, 1.0 = 100%)
# Acceptable range: 0.30-0.60 (higher = fewer but more accurate predictions)
MIN_CONFIDENCE_THRESHOLD = 0.40

# Haar Cascade scale factor: How much the image size is reduced at each scale
# Smaller values (closer to 1.0) are more thorough but slower
# Acceptable range: 1.05-1.30 (lower = more accurate but slower detection)
SCALE_FACTOR = 1.15

# Haar Cascade min neighbors: Minimum number of neighbor detections for valid face
# Higher values reduce false positives but may miss some faces
# Acceptable range: 3-6 (lower = more sensitive, higher = more strict)
MIN_NEIGHBORS = 4


def build_model(input_shape=(48, 48, 1), num_classes=7):
    """Build and compile the CNN emotion classifier."""
    model = Sequential(
        [
            Input(shape=input_shape),

            Conv2D(32, (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            Activation("relu"),

            Conv2D(64, (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),

            Conv2D(128, (3, 3), padding="same", use_bias=False),
            BatchNormalization(),
            Activation("relu"),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),

            Flatten(),
            Dense(256, use_bias=False),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.5),

            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def validate_dataset_classes(train_generator, validation_generator):
    """Ensure training and validation class indices have identical meanings."""
    train_classes = train_generator.class_indices
    valid_classes = validation_generator.class_indices

    if train_classes != valid_classes:
        raise ValueError(
            "Training and validation class folders do not match.\n"
            f"Training classes: {train_classes}\n"
            f"Validation classes: {valid_classes}"
        )

    actual_classes = list(train_classes.keys())
    if actual_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Dataset class folders must be exactly these seven names: "
            f"{EXPECTED_CLASSES}\nFound: {actual_classes}"
        )

    if len(train_classes) != NUM_CLASSES:
        raise ValueError(
            f"Expected {NUM_CLASSES} emotion classes, found {len(train_classes)}."
        )


def train_model():
    """Create generators, train the model, and save the best weights."""
    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALID_DIR):
        raise FileNotFoundError(
            "Training dataset was not found.\n"
            f"Expected training folder: {TRAIN_DIR}\n"
            f"Expected validation folder: {VALID_DIR}\n"
            "Create these folders or provide a trained model file."
        )

    train_data_gen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )
    validation_data_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_data_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=True,
    )
    validation_generator = validation_data_gen.flow_from_directory(
        VALID_DIR,
        target_size=TARGET_SIZE,
        batch_size=BATCH_SIZE,
        color_mode="grayscale",
        class_mode="categorical",
        shuffle=False,
    )

    if train_generator.samples == 0 or validation_generator.samples == 0:
        raise ValueError("The training and validation folders must contain images.")

    validate_dataset_classes(train_generator, validation_generator)

    model = build_model(
        input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 1),
        num_classes=NUM_CLASSES,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_FILENAME,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    print("Starting model training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=callbacks,
    )

    # EarlyStopping restores the best in-memory weights. Save those weights too.
    model.save(MODEL_FILENAME)
    print(f"Best model saved to: {MODEL_FILENAME}")

    return model, history


def load_or_train_model():
    """Load a saved model, or train one if no saved model exists."""
    if os.path.isfile(MODEL_FILENAME):
        print(f"Loading existing model: {MODEL_FILENAME}")
        try:
            model = load_model(MODEL_FILENAME)
            print("Model loaded successfully.")
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Could not load model '{MODEL_FILENAME}'. "
                "Check that the file is valid and compatible with your TensorFlow/Keras version. "
                "If you intend to retrain, move or rename the existing model file first."
            ) from exc

    print("No saved model found. Checking for training data...")
    model, _ = train_model()
    return model


def run_webcam(model):
    """Run face detection and emotion prediction until the user presses q."""
    face_cascade_path = (
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(face_cascade_path)

    if face_cascade.empty():
        raise RuntimeError(f"Could not load Haar cascade: {face_cascade_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not open webcam. Check camera permissions or try another camera index."
        )

    print("Webcam started. Press 'q' to quit.")

    frame_count = 0  # Start at 0 for proper modulo arithmetic
    last_predictions = {}  # Cache predictions for skipped frames: {(x,y,w,h): (display_text, box_color)}

    try:
        while True:
            # Measure frame time at the START of each iteration for accurate FPS control
            loop_start_time = time.time()
            
            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read a frame from the webcam.")
                break

            # Process face detection and prediction only on selected frames
            if frame_count % (FRAME_SKIP + 1) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=(30, 30),
                )

                # Update cache with new predictions (don't clear - let old predictions persist if no faces detected)
                new_predictions = {}

                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y + h, x:x + w]

                    try:
                        face = cv2.resize(roi_gray, TARGET_SIZE)
                        face = face.astype("float32") / 255.0
                        face = np.expand_dims(face, axis=-1)
                        face = np.expand_dims(face, axis=0)

                        predictions = model.predict(face, verbose=0)[0]

                        if predictions.size != NUM_CLASSES:
                            raise ValueError(
                                f"Model returned {predictions.size} classes; "
                                f"expected {NUM_CLASSES}."
                            )

                        class_index = int(np.argmax(predictions))
                        confidence = float(predictions[class_index]) * 100.0
                        
                        # Apply confidence threshold filtering
                        if confidence >= (MIN_CONFIDENCE_THRESHOLD * 100):
                            # High confidence: display emotion label with confidence
                            emotion = EMOTION_LABELS.get(class_index, "Unknown")
                            display_text = f"{emotion} ({confidence:.1f}%)"
                            box_color = (0, 255, 0)  # Green for high confidence
                        else:
                            # Low confidence: display "Uncertain" label
                            display_text = "Uncertain"
                            box_color = (0, 165, 255)  # Orange for uncertain

                    except Exception as exc:
                        print(f"Prediction error: {exc}")
                        display_text = "Prediction error"
                        box_color = (0, 0, 255)  # Red for errors

                    # Store the new prediction for this face location
                    new_predictions[(x, y, w, h)] = (display_text, box_color)

                # Replace cache with new predictions from this frame
                last_predictions = new_predictions

            # Draw all cached predictions (works for both processing and skipped frames)
            for (x, y, w, h), (display_text, box_color) in last_predictions.items():
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2,
                )

                text_y = y - 10 if y - 10 > 20 else y + h + 25
                cv2.putText(
                    frame,
                    display_text,
                    (x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Emotion Detection", frame)

            # Frame rate control: calculate required delay to maintain TARGET_FPS
            loop_time = time.time() - loop_start_time
            required_delay = max(1, int((1000.0 / TARGET_FPS) - (loop_time * 1000)))
            
            if cv2.waitKey(required_delay) & 0xFF == ord("q"):
                break
            
            frame_count += 1  # Increment after processing

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    try:
        model = load_or_train_model()
        run_webcam(model)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nProgram stopped by user.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
