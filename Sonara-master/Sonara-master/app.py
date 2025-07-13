from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
# from driverCode import mediapipe_detection
import os
# import time
import keras
# from scipy import stats

# Disable GPU if CUDA is available
# if "NVIDIA_VISIBLE_DEVICES" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ:
#     os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#     print("GPU detected and disabled.")
# else:
#     print("No GPU detected. Running on CPU.")

app = Flask(__name__)

mp_holistic = mp.solutions.holistic
# holistic = mp_holistic.Holistic()
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_face_mesh = mp.solutions.face_mesh  # Import face_mesh for facial landmarks

# --- Load Keras Model ---
# Ensure the model file is in the same directory or provide the correct path
try:
    model = keras.models.load_model("reallylatest2.keras")
    # print("Model loaded successfully.")
    # model.summary() # Optional: print model summary
except Exception as e:
    print(f"Error loading Keras model: {e}")
    # Handle the error appropriately, maybe exit or use a fallback
    exit()

# --- Constants and Global Variables ---
actions = np.array(['hello', 'thanks', 'iloveyou']) # Actions to detect
sequence_length = 30
threshold = 0.5 # Confidence threshold
colors = [(245,117,16), (117,245,16), (16,117,245)] * (len(actions) // 3 + 1) # Visualization colors

# Variables for prediction logic
sequence = []
sentence = []
predictions = []

# Add this near the other global variables
camera = None

# Add this to control camera release
def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None
        print("Camera released.")

# --- Helper Functions (Adapted from driverCode.py) ---
def mediapipe_detection(image, model):
    """Processes an image frame to detect holistic landmarks."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    """Draws styled landmarks on the image."""
    # Draw face connections
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image,
            results.face_landmarks,
            mp_face_mesh.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    # Draw pose connections
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
    # Draw left hand connections
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
    # Draw right hand connections
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

def extract_keypoints(results):
    """Extracts keypoints from MediaPipe results."""
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    """Visualizes prediction probabilities on the frame."""
    output_frame = input_frame.copy()
    num_actions = min(len(res), len(actions))

    for num in range(num_actions):
        prob = res[num]
        # print(f"Action: {actions[num]}, Probability: {prob:.4f}") # Debug print
        color = colors[num % len(colors)]
        # Draw rectangle based on probability
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        # Put text label for the action
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return output_frame

# --- Video Generation Function ---
def generate_frames():
    """Generates frames for the video stream."""
    global sequence, sentence, predictions, camera # Use global variables

    # Initialize camera
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video capture device.")
        return

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while camera is not None and camera.isOpened():  # Check if camera is still valid
            # Read feed
            ret, frame = camera.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break # Exit loop if frame capture fails

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-sequence_length:] # Keep last 30 frames

            res = [0.0] * len(actions) # Default probabilities if sequence not full

            if len(sequence) == sequence_length:
                try:
                    prediction_result = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                    res = prediction_result # Get probabilities
                    # print(actions[np.argmax(res)]) # Optional: Print predicted action to console
                    predictions.append(np.argmax(res))

                    # Visualization logic (sentence building)
                    # Check if the most dominant prediction in the last 10 frames is the current one
                    if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            current_action = actions[np.argmax(res)]
                            if not sentence or sentence[-1] != current_action:
                                sentence.append(current_action)

                    # Limit sentence length
                    if len(sentence) > 5:
                        sentence = sentence[-5:]
                except Exception as e:
                    print(f"Error during prediction or post-processing: {e}")
                    # Reset or handle error state if necessary
                    # sequence = [] # Optional: Reset sequence on error

            # Viz probabilities on the frame
            # Ensure 'res' is a list or array of probabilities
            if isinstance(res, (list, np.ndarray)) and len(res) == len(actions):
                 image = prob_viz(res, actions, image, colors)
            else:
                 print(f"Warning: 'res' is not in the expected format. Type: {type(res)}, Value: {res}")
                 # Handle unexpected 'res' format, maybe draw default visualization


            # Draw sentence bar
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Encode frame as JPEG
            try:
                ret, buffer = cv2.imencode('.jpg', image)
                if not ret:
                    print("Error: Failed to encode frame.")
                    continue # Skip this frame if encoding fails
                frame_bytes = buffer.tobytes()
                # Yield the frame in the required format for multipart response
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                print(f"Error encoding or yielding frame: {e}")
                # Consider breaking or specific error handling

    # Release the camera if it's still open
    release_camera()

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Provides the video stream."""
    # Ensure globals are reset if needed upon new connection, or manage state per user session if required
    # For this simple example, we rely on the single global state
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Add this new route to handle camera stopping
@app.route('/stop_camera')
def stop_camera():
    """Stop and release the camera."""
    release_camera()
    return jsonify({"status": "success", "message": "Camera released"})

# --- Main Execution ---
import os

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if PORT is not set
    app.run(debug=False, host='0.0.0.0', port=port)