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
# Frame rate control: Target maximum frames per second for video display.
# All detected faces are classified together in one batch on every processed frame.
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
    """Detect every visible face and classify all faces in one batch per frame."""
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

    # Request a practical webcam resolution. The camera may choose a different size.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam started. Press 'q' to quit.")
    print("Multiple faces are classified together. Keep faces well-lit and facing the camera.")

    try:
        while True:
            loop_start_time = time.time()

            ret, frame = cap.read()
            if not ret or frame is None:
                print("Failed to read a frame from the webcam.")
                break

            # Mirror the preview so it behaves like a normal selfie camera.
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detectMultiScale returns all face rectangles it can find, not just one.
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=SCALE_FACTOR,
                minNeighbors=MIN_NEIGHBORS,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            face_boxes = []
            face_batch = []

            # Prepare every detected face for the CNN.
            for (x, y, w, h) in faces:
                x, y, w, h = int(x), int(y), int(w), int(h)
                roi_gray = gray[y:y + h, x:x + w]

                if roi_gray.size == 0:
                    continue

                face_image = cv2.resize(
                    roi_gray,
                    TARGET_SIZE,
                    interpolation=cv2.INTER_AREA,
                )
                face_image = face_image.astype(np.float32) / 255.0
                face_image = np.expand_dims(face_image, axis=-1)  # (48, 48, 1)

                face_boxes.append((x, y, w, h))
                face_batch.append(face_image)

            face_results = []

            if face_batch:
                # IMPORTANT: one model call for all detected faces.
                # Input shape: (number_of_faces, 48, 48, 1)
                batch_array = np.asarray(face_batch, dtype=np.float32)

                try:
                    batch_predictions = model.predict(
                        batch_array,
                        batch_size=min(BATCH_SIZE, len(face_batch)),
                        verbose=0,
                    )

                    if batch_predictions.shape != (len(face_batch), NUM_CLASSES):
                        raise ValueError(
                            "Unexpected model output shape. "
                            f"Received {batch_predictions.shape}; expected "
                            f"({len(face_batch)}, {NUM_CLASSES})."
                        )

                    for prediction in batch_predictions:
                        class_index = int(np.argmax(prediction))
                        confidence = float(prediction[class_index])

                        if not np.isfinite(confidence):
                            face_results.append(("Uncertain", (0, 165, 255)))
                        elif confidence >= MIN_CONFIDENCE_THRESHOLD:
                            emotion = EMOTION_LABELS.get(class_index, "Unknown")
                            label = f"{emotion} ({confidence * 100:.1f}%)"
                            face_results.append((label, (0, 255, 0)))
                        else:
                            face_results.append(("Uncertain", (0, 165, 255)))

                except Exception as exc:
                    print(f"Batch prediction error: {exc}")
                    face_results = [
                        ("Prediction error", (0, 0, 255))
                        for _ in face_boxes
                    ]

            # Draw one rectangle and one emotion label for each detected face.
            for (x, y, w, h), (label, box_color) in zip(face_boxes, face_results):
                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + w, y + h),
                    box_color,
                    2,
                )

                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Keep the label inside the frame, even for faces near the top edge.
                text_x = max(0, x)
                text_y = y - 8 if y - text_height - 12 >= 0 else y + h + text_height + 8
                text_y = min(text_y, frame.shape[0] - baseline - 2)

                background_top = max(0, text_y - text_height - 8)
                background_bottom = min(frame.shape[0], text_y + baseline + 2)
                background_right = min(frame.shape[1], text_x + text_width + 8)
                cv2.rectangle(
                    frame,
                    (text_x, background_top),
                    (background_right, background_bottom),
                    box_color,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (text_x + 4, text_y - 2),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

            # Show the current number of detected faces.
            cv2.putText(
                frame,
                f"Faces detected: {len(face_boxes)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Multiple Face Emotion Detection", frame)

            # Limit display rate without intentionally skipping face detection.
            elapsed = time.time() - loop_start_time
            delay_ms = max(1, int((1000.0 / TARGET_FPS) - (elapsed * 1000.0)))
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                break

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
