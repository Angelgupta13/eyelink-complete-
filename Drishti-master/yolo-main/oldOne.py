import cv2
import numpy as np

# Initialize YOLO parameters
whT = 320
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

# Function to detect objects in the frame
def find_objects(outputs, frame):
    hT, wT, cT = frame.shape
    bbox, classIds, confs, detected_objects = [], [], [], []

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
            label = f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'
            detected_objects.append(label)

            # Draw bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    print("Detected objects:", detected_objects)
    print(f"Number of detected objects: {len(detected_objects)}")  # Display the number of detected objects
    print("------")
    # Display the detected objects

    # Add a semi-transparent overlay for the transcript
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (300, 20 + 20 * len(detected_objects)), (0, 0, 0), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # Display detected objects as a transcript
    y_offset = 20
    for obj in detected_objects:
        cv2.putText(frame, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 20

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
        find_objects(outputs, frame)

        # Display the frame with detections
        cv2.imshow('Object Detection', frame)

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