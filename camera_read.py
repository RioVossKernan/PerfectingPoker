import cv2
import tensorflow as tf
import numpy as np
import nbimporter
from dual_model import CardPredictor

# ----------------------------------------------------------------------
# Load the pre-trained models

suit_model = CardPredictor(4)
suit_model.build((None, 300, 300, 3))
suit_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
suit_model.load_weights("suit_model_ckpt.keras")


rank_model = CardPredictor(13)
rank_model.build((None, 300, 300, 3))
rank_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
rank_model.load_weights("rank_model_ckpt.keras")

# ----------------------------------------------------------------------

def center_crop_and_resize(image, target_size=(300, 300)):
    h, w = image.shape[:2]
    
    # Find the shorter dimension and crop to a centered square
    min_dim = min(h, w)
    start_x = (w - min_dim) // 2
    start_y = (h - min_dim) // 2
    cropped = image[start_y:start_y + min_dim, start_x:start_x + min_dim]

    # Resize to the target size (e.g., 300x300)
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

# ----------------------------------------------------------------------

# Open the default camera (usually camera 0)
#cam = cv2.VideoCapture(2) # This is the index of the Iphone's camera, specific for Member2's Iphone, If you are testing with different Cameras, chaneg the index
cam = cv2.VideoCapture(0)

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
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('c'):
        print("Capturing frame...")

        # Preprocess the frame
        image = tf.convert_to_tensor(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image = center_crop_and_resize(image.numpy(), target_size=(300, 300))
        image = tf.cast(image, tf.float32) / 255.0
        image = np.expand_dims(image, axis=0)

        # Predict
        suit_n = tf.argmax(suit_model.predict(image), axis=1).numpy()[0]
        rank_n = tf.argmax(rank_model.predict(image), axis=1).numpy()[0]
        
        suit_name = suits[suit_n]
        rank_name = ranks[rank_n]

        # Overlay prediction and display result on the top right corner of the image. 
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Prediction: {rank_name} of {suit_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Prediction", display_frame)
        cv2.waitKey(1500)  # Show prediction for 1.5 seconds

# Release the camera and close any OpenCV windows
cam.release()
cv2.destroyAllWindows()