import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# The folder containing this tracker.py file.
PROJECT_DIR = Path(__file__).resolve().parent

# Required MediaPipe model file.
MODEL_FILE = PROJECT_DIR / "hand_landmarker.task"

# Recorded hand movement will be saved here.
RECORDINGS_DIR = PROJECT_DIR / "recordings"
OUTPUT_FILE = RECORDINGS_DIR / "hand_recording.json"


# MediaPipe hand landmark connections.
HAND_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


def get_hand_label_and_score(result):
    """
    MediaPipe returns handedness in this structure:

        result.handedness[hand_index][category_index]

    For one detected hand, the first category contains the label
    and confidence score.
    """

    if not result.handedness:
        return "Unknown", 0.0

    if len(result.handedness[0]) == 0:
        return "Unknown", 0.0

    category = result.handedness[0][0]

    label = getattr(category, "category_name", None)
    if not label:
        label = getattr(category, "display_name", None)
    if not label:
        label = "Unknown"

    score = getattr(category, "score", 0.0)
    if score is None:
        score = 0.0

    return label, float(score)


def extract_hand(result):
    """
    Convert the first detected hand into the JSON format used by
    this project.
    """

    if not result.hand_landmarks:
        return None

    hand_landmarks = result.hand_landmarks[0]
    label, score = get_hand_label_and_score(result)

    return {
        "label": label,
        "score": score,
        "landmarks": [
            {
                "x": float(landmark.x),
                "y": float(landmark.y),
                "z": float(landmark.z),
            }
            for landmark in hand_landmarks
        ],
    }


def draw_hand(frame, result):
    """
    Draw the detected hand landmarks and connections on the camera frame.
    """

    if not result.hand_landmarks:
        return

    height, width, _ = frame.shape
    hand_landmarks = result.hand_landmarks[0]

    for start_index, end_index in HAND_CONNECTIONS:
        start = hand_landmarks[start_index]
        end = hand_landmarks[end_index]

        start_point = (
            int(start.x * width),
            int(start.y * height),
        )

        end_point = (
            int(end.x * width),
            int(end.y * height),
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (0, 255, 0),
            2,
        )

    for landmark in hand_landmarks:
        point = (
            int(landmark.x * width),
            int(landmark.y * height),
        )

        cv2.circle(
            frame,
            point,
            5,
            (0, 0, 255),
            -1,
        )


def save_recording(frames):
    """
    Save the captured frames to recordings/hand_recording.json.
    """

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    recording = {
        "format": "hand-motion-prototype-v1",
        "landmark_count": 21,
        "created_at": time.time(),
        "frames": frames,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(recording, file, indent=2)

    print()
    print(f"Saved {len(frames)} frames.")
    print(f"Recording file:")
    print(OUTPUT_FILE)


def main():
    print(f"Project folder: {PROJECT_DIR}")
    print(f"Model file:     {MODEL_FILE}")
    print(f"Output file:    {OUTPUT_FILE}")
    print()

    if not MODEL_FILE.is_file():
        print("ERROR: The MediaPipe model file was not found.")
        print()
        print("Expected file:")
        print(MODEL_FILE)
        print()
        print("Make sure the file is named exactly:")
        print("HandLandmarker.task")
        return

    RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("ERROR: Could not open the webcam.")
        print("Check that another program is not using the camera.")
        return

    recording = False
    frames = []
    recording_start_time = None

    try:
        base_options = python.BaseOptions(
            model_asset_path=str(MODEL_FILE)
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        detector = vision.HandLandmarker.create_from_options(options)

        print("Tracker started.")
        print("Press R in the camera window to start recording.")
        print("Press R again to stop and save.")
        print("Press Q to quit.")
        print()

        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a frame from the webcam.")
                break

            # Mirror the camera preview.
            frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame,
            )

            result = detector.detect(mp_image)

            detected_hand = extract_hand(result)

            draw_hand(frame, result)

            if recording:
                elapsed_time = (
                    time.perf_counter() - recording_start_time
                )

                frames.append(
                    {
                        "time": elapsed_time,
                        "hand": detected_hand,
                    }
                )

            if recording:
                status = "RECORDING"
                color = (0, 0, 255)
            else:
                status = "READY"
                color = (0, 255, 0)

            cv2.putText(
                frame,
                f"{status} | R: record/stop | Q: quit",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
            )

            if detected_hand is None:
                cv2.putText(
                    frame,
                    "No hand detected",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2,
                )
            else:
                label = detected_hand["label"]
                score = detected_hand["score"]

                cv2.putText(
                    frame,
                    f"{label} hand ({score:.2f})",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("Hand Motion Capture", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("r"):
                if not recording:
                    frames = []
                    recording_start_time = time.perf_counter()
                    recording = True
                    print("Recording started.")
                else:
                    recording = False
                    save_recording(frames)
                    print("Recording stopped.")

            elif key == ord("q"):
                if recording:
                    recording = False
                    save_recording(frames)

                print("Exiting.")
                break

        detector.close()

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()