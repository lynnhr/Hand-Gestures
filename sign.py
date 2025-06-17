import cv2
import mediapipe as mp
from joblib import load
import numpy as np
from scipy.stats import mode

def preprocess_landmarks(landmark_list, frame_width, frame_height):
    wrist = landmark_list[0]
    mid_tip = landmark_list[12]
    wrist_x, wrist_y = wrist.x * frame_width, wrist.y * frame_height
    mid_tip_x, mid_tip_y = mid_tip.x * frame_width, mid_tip.y * frame_height

    processed = []
    for lm in landmark_list:
        x = (lm.x * frame_width - wrist_x) / (mid_tip_x - wrist_x + 1e-6)
        y = (lm.y * frame_height - wrist_y) / (mid_tip_y - wrist_y + 1e-6)
        processed.extend([x, y, lm.z])
    return processed

# Load model
model_data = load('hand_gesture_model_package.joblib')
model = model_data['model']
le = model_data['label_encoder']

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
prediction_history = []
window_size = 5  # Number of frames to consider for mode calculation

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    # Process frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            try:
                # Preprocess and predict
                landmarks = hand_landmarks.landmark
                processed = preprocess_landmarks(landmarks, frame.shape[1], frame.shape[0])
                prediction = model.predict([processed])[0]

                # Update prediction history
                prediction_history.append(prediction)
                if len(prediction_history) > window_size:
                    prediction_history.pop(0)

                # Robust mode calculation
                if len(prediction_history) > 0:
                    mode_result = mode(prediction_history)
                    current_prediction = mode_result.mode[0] if isinstance(mode_result.mode, np.ndarray) else mode_result.mode
                    gesture_name = le.inverse_transform([current_prediction])[0]

                    # Display results
                    cv2.putText(frame, f'Gesture: {gesture_name}', (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Draw landmarks
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            except Exception as e:
                print(f"Processing error: {e}")
                continue

    cv2.imshow('Hand Gesture Recognition', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()