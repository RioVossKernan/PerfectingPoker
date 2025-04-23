import cv2

# Open the default camera (usually camera 0)
cam = cv2.VideoCapture(3) 


# Check if the camera opened successfully
if not cam.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read a frame from the camera
    ret, frame = cam.read()

    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Display the frame in a window
    cv2.imshow('Camera Feed', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release the camera and close any OpenCV windows
cam.release()
cv2.destroyAllWindows()