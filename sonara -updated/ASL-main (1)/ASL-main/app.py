from flask import Flask, request, jsonify, render_template, Response
import cv2
import numpy as np
import mediapipe as mp
import os
from tensorflow import keras
import base64

# Check if CUDA is available
if "NVIDIA_VISIBLE_DEVICES" in os.environ or "CUDA_VISIBLE_DEVICES" in os.environ:
    print("GPU detected and ENABLED.")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("No GPU detected. Running on CPU.")

app = Flask(__name__)

# --- MediaPipe and Model Setup ---
mp_holistic_solution = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize MediaPipe Holistic model ONCE
holistic_model = mp_holistic_solution.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

try:
    model = keras.models.load_model("reallylatest2.keras")
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading Keras model: {e}")
    exit()

# --- Constants and Global Variables ---
actions = np.array(['hello', 'thanks', 'iloveyou'])
sequence_length = 30
threshold = 0.5
colors = [(245,117,16), (117,245,16), (16,117,245)] * (len(actions) // 3 + 1)

sequence = []
sentence = []
predictions = []

# --- Helper Functions ---
def mediapipe_detection(image, holistic_processor): # Renamed model to holistic_processor for clarity
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic_processor.process(image_rgb)
    image_rgb.flags.writeable = True
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    return image_bgr, results

def draw_styled_landmarks(image, results):
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic_solution.FACEMESH_TESSELATION, # Corrected constant
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic_solution.POSE_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic_solution.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic_solution.HAND_CONNECTIONS,
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3) # Using mp_holistic_solution.FACEMESH_NUM_LANDMARKS might be more robust
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions_arr, input_frame, viz_colors): # Renamed for clarity
    output_frame = input_frame.copy()
    num_actions_to_viz = min(len(res), len(actions_arr))
    for num in range(num_actions_to_viz):
        prob = res[num]
        color_val = viz_colors[num % len(viz_colors)]
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color_val, -1)
        cv2.putText(output_frame, actions_arr[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# --- Frame Processing Function ---
def process_single_frame(frame_to_process): # Renamed from generate_frames
    global sequence, sentence, predictions, holistic_model # Use the global holistic_model

    image, results = mediapipe_detection(frame_to_process, holistic_model) # Pass the global model
    draw_styled_landmarks(image, results)

    keypoints = extract_keypoints(results)
    sequence.append(keypoints)
    sequence = sequence[-sequence_length:]

    res_probs = [0.0] * len(actions)

    if len(sequence) == sequence_length:
        try:
            prediction_result = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            res_probs = prediction_result
            predictions.append(np.argmax(res_probs))

            if len(predictions) >= 10 and np.unique(predictions[-10:])[0] == np.argmax(res_probs):
                if res_probs[np.argmax(res_probs)] > threshold:
                    current_action = actions[np.argmax(res_probs)]
                    if not sentence or sentence[-1] != current_action:
                        sentence.append(current_action)
            if len(sentence) > 5:
                sentence = sentence[-5:]
        except Exception as e:
            print(f"Error during prediction: {e}")
            # Consider resetting sequence or predictions here if error is persistent

    if isinstance(res_probs, (list, np.ndarray)) and len(res_probs) == len(actions):
        image = prob_viz(res_probs, actions, image, colors)
    else:
        print(f"Warning: 'res_probs' is not in the expected format. Type: {type(res_probs)}, Value: {res_probs}")

    cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
    cv2.putText(image, ' '.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return image

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def handle_process_frame():
    try:
        # Check if the frame is sent as a file (binary data)
        if 'frame' in request.files:
            file = request.files['frame']
            np_img = np.frombuffer(file.read(), np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        # Fallback to JSON Base64 data if not sent as file
        elif request.is_json:
            data = request.get_json()
            frame_data = data['frame']
            encoded_data = frame_data.split(',')[1]
            decoded_data = base64.b64decode(encoded_data)
            np_data = np.frombuffer(decoded_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
        else:
            return jsonify({'error': 'Invalid frame data format'}), 400

        if frame is None:
            return jsonify({'error': 'Could not decode frame'}), 400

        processed_image = process_single_frame(frame) # Use the renamed function

        _, buffer = cv2.imencode('.jpg', processed_image)
        processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'processed_frame': f'data:image/jpeg;base64,{processed_frame_base64}'})
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Waitress server on http://localhost:5000/")
    serve(app, host='0.0.0.0', port=port, threads=8) # Adjust threads as needed
