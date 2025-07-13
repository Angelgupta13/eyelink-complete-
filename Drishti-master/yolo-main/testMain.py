import cv2
import numpy as np
import pyttsx3

# Initialize YOLO parameters
whT = 320 # Width and height for resizing the input image, must be divisible by 32, affects the model's performance and speed tradeoff
confThreshold = 0.5
nmsThreshold = 0.3

# Load class names from the COCO dataset
def load_class_names(file_path):
    with open(file_path, 'rt') as f:
        return f.read().rstrip('\n').split('\n')

classNames = load_class_names('coco.names')

# Load YOLO model
def load_yolo_model(config, weights):
    net = cv2.dnn.readNetFromDarknet(config, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

net = load_yolo_model('yolov3.cfg', 'yolov3.weights')

# Function to detect objects in the frame and estimate distance
def find_objects(outputs, frame):
    hT, wT, cT = frame.shape
    bbox, classIds, confs, detected_objects = [], [], [], []
    transcript = []

    # Define real-world height of the object (in meters) for known classes
    real_heights = {
        'person': 1.7,  # Average height of a person in meters
        'bicycle': 1.0,  # Average height of a bicycle
        'car': 1.5,  # Average height of a car
        'motorbike': 1.1,  # Average height of a motorbike
        'aeroplane': 3.5,  # Average height of an aeroplane
        'bus': 3.0,  # Average height of a bus
        'train': 3.5,  # Average height of a train
        'truck': 3.0,  # Average height of a truck
        'boat': 2.0,  # Average height of a boat
        'traffic light': 2.5,  # Average height of a traffic light
        'fire hydrant': 0.8,  # Average height of a fire hydrant
        'stop sign': 1.2,  # Average height of a stop sign
        'parking meter': 1.5,  # Average height of a parking meter
        'bench': 0.5,  # Average height of a bench
        'bird': 0.3,  # Average height of a bird
        'cat': 0.4,  # Average height of a cat
        'dog': 0.5,  # Average height of a dog
        'horse': 1.6,  # Average height of a horse
        'sheep': 1.0,  # Average height of a sheep
        'cow': 1.5,  # Average height of a cow
        'elephant': 3.0,  # Average height of an elephant
        'bear': 2.0,  # Average height of a bear
        'zebra': 1.5,  # Average height of a zebra
        'giraffe': 5.0,  # Average height of a giraffe
        'backpack': 0.5,  # Average height of a backpack
        'umbrella': 1.0,  # Average height of an umbrella
        'handbag': 0.4,  # Average height of a handbag
        'tie': 0.3,  # Average height of a tie
        'suitcase': 0.7,  # Average height of a suitcase
        'frisbee': 0.3,  # Average height of a frisbee
        'skis': 1.8,  # Average height of skis
        'snowboard': 1.5,  # Average height of a snowboard
        'sports ball': 0.2,  # Average height of a sports ball
        'kite': 1.0,  # Average height of a kite
        'baseball bat': 1.0,  # Average height of a baseball bat
        'baseball glove': 0.3,  # Average height of a baseball glove
        'skateboard': 0.8,  # Average height of a skateboard
        'surfboard': 2.0,  # Average height of a surfboard
        'tennis racket': 0.7,  # Average height of a tennis racket
        'bottle': 0.3,  # Average height of a bottle
        'wine glass': 0.2,  # Average height of a wine glass
        'cup': 0.15,  # Average height of a cup
        'fork': 0.2,  # Average height of a fork
        'knife': 0.25,  # Average height of a knife
        'spoon': 0.2,  # Average height of a spoon
        'bowl': 0.2,  # Average height of a bowl
        'banana': 0.2,  # Average height of a banana
        'apple': 0.1,  # Average height of an apple
        'sandwich': 0.1,  # Average height of a sandwich
        'orange': 0.1,  # Average height of an orange
        'broccoli': 0.2,  # Average height of broccoli
        'carrot': 0.2,  # Average height of a carrot
        'hot dog': 0.2,  # Average height of a hot dog
        'pizza': 0.3,  # Average height of a pizza
        'donut': 0.1,  # Average height of a donut
        'cake': 0.2,  # Average height of a cake
        'chair': 1.0,  # Average height of a chair
        'sofa': 1.0,  # Average height of a sofa
        'pottedplant': 0.5,  # Average height of a potted plant
        'bed': 0.8,  # Average height of a bed
        'diningtable': 0.8,  # Average height of a dining table
        'toilet': 0.7,  # Average height of a toilet
        'tvmonitor': 0.6,  # Average height of a TV monitor
        'laptop': 0.3,  # Average height of a laptop
        'mouse': 0.1,  # Average height of a mouse
        'remote': 0.2,  # Average height of a remote
        'keyboard': 0.2,  # Average height of a keyboard
        'cell phone': 0.15,  # Average height of a cell phone
        'microwave': 0.4,  # Average height of a microwave
        'oven': 0.6,  # Average height of an oven
        'toaster': 0.3,  # Average height of a toaster
        'sink': 0.5,  # Average height of a sink
        'refrigerator': 1.8,  # Average height of a refrigerator
        'book': 0.3,  # Average height of a book
        'clock': 0.3,  # Average height of a clock
        'vase': 0.4,  # Average height of a vase
        'scissors': 0.2,  # Average height of scissors
        'teddy bear': 0.5,  # Average height of a teddy bear
        'hair drier': 0.3,  # Average height of a hair drier
        'toothbrush': 0.2  # Average height of a toothbrush
    }

    # Focal length (calculated using a reference object during calibration)
    focal_length = 35  # Example value, adjust based on your camera
    # focal_length = 615  # Example value, adjust based on your camera

    # Parse YOLO outputs
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    # Apply Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = bbox[i]
            label = classNames[classIds[i]].lower()
            detected_objects.append(label)
            
            # Estimate distance if the object's real height is known
            if label in real_heights:
                distance = (real_heights[label] * focal_length) / h
                distance_text = f'{label.upper()} {int(confs[i] * 100)}% {distance:.2f}m'
            else:
                distance_text = f'{label.upper()} {int(confs[i] * 100)}%'
            print("Detected objects:", distance_text)
            # Draw bounding box and label with distance
            transcript.append(distance_text)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, distance_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
        print("---------------------------")
    # Add a semi-transparent overlay for the transcript
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 20 + 20 * len(detected_objects)), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display detected objects as a transcript
    y_offset = 20
    for obj in detected_objects:
        cv2.putText(frame, distance_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20
    return '. '.join(transcript)

# Function to preprocess the frame for YOLO
def preprocess_frame(frame):
    blob = cv2.dnn.blobFromImage(frame, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[int(i) - 1] for i in net.getUnconnectedOutLayers()]
    return net.forward(outputNames)

# Main function to run object detection
def main():
    # Initialize the local camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Adjust speech rate
    tts_engine.setProperty('volume', 1.0)  # Set volume (0.0 to 1.0)

    # Set a window title and allow resizing
    cv2.namedWindow('Object Detection', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Object Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame from the local camera")
            break

        # Process the frame and detect objects
        outputs = preprocess_frame(frame)
        transcript = find_objects(outputs, frame)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

        # Speak the transcript
        if transcript:
            tts_engine.say(transcript)
            tts_engine.runAndWait()

        # Exit the loop when 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('p'):
            print("Paused. Press 'p' again to resume.")
            while cv2.waitKey(1) & 0xFF != ord('p'):
                pass
        elif key == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
