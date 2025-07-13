import cv2
import numpy as np

# Initialize the local camera (camera index 0)
cap = cv2.VideoCapture(0)

# Set parameters for YOLO
whT = 320
confThreshold = 0.5
nmsThreshold = 0.3

# Load class names from the COCO dataset
classesfile = 'coco.names'
classNames = []
with open(classesfile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Load YOLO model configuration and weights
modelConfig = 'yolov3.cfg'
modelWeights = 'yolov3.weights'
net = cv2.dnn.readNetFromDarknet(modelConfig, modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Function to find objects in the frame
def findObject(outputs, im):
    hT, wT, cT = im.shape
    bbox = []
    classIds = []
    confs = []
    detected_objects = []  # List to store detected objects

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
    
    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)
    if len(indices) > 0:
        for i in indices.flatten():
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            label = f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%'
            detected_objects.append(label)  # Add detected object to the list

            # Draw bounding box and label on the frame
            cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(im, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    # Display the transcript of detected objects on the frame
    y_offset = 20
    for obj in detected_objects:
        cv2.putText(im, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 20

# Create a named window and set it to full screen
cv2.namedWindow('Image', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
# alt tab to switch eindow

# Main loop to process video frames
while True:
    # Capture frame from the local camera
    success, im = cap.read()
    if not success:
        print("Failed to grab frame from the local camera")
        break

    # Preprocess the frame for the YOLO model
    blob = cv2.dnn.blobFromImage(im, 1/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layernames = net.getLayerNames()
    outputNames = [layernames[int(i) - 1] for i in net.getUnconnectedOutLayers()]

    # Perform forward pass to get outputs
    outputs = net.forward(outputNames)

    # Detect objects in the frame
    findObject(outputs, im)

    # Display the frame with detections
    cv2.imshow('Image', im)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()