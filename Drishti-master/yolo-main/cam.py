# import cv2

# # Use the default local camera (camera index 0)
# cap = cv2.VideoCapture(0)

# # Check if the local camera is opened successfully
# if not cap.isOpened():
#     print("Failed to open the local camera")
#     exit()

# # Create a named window for displaying the footage
# cv2.namedWindow("Live Cam Testing", cv2.WINDOW_AUTOSIZE)

# # Read and display video frames
# while True:
#     # Read a frame from the video stream
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

#     # Display the frame
#     cv2.imshow('Live Cam Testing', frame)

#     # Exit the loop when 'q' is pressed
#     key = cv2.waitKey(5)
#     if key == ord('q'):
#         break

# # Release the VideoCapture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()