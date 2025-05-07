import cv2
import tensorflow as tf
import numpy as np
import nbimporter
from cropped_img_model import CardModel, crop_and_warp

# ----------------------------------------------------------------------
# Load the pre-trained models

suit_model = CardModel(4)
suit_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
suit_model.build(input_shape=(1, 80, 60, 3))
suit_model.load_weights("cropped_suit_model.weights.h5")


rank_model = CardModel(13)
rank_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
rank_model.build(input_shape=(1, 80, 60, 3))
rank_model.load_weights("cropped_rank_model.weights.h5")

# ----------------------------------------------------------------------


ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['Hearts', 'Diamonds', 'Spades', 'Clubs']

def load_and_preprocess_frame(frame):
    img = crop_and_warp(frame) # crop and warp
    img = cv2.resize(img, (250, 350)) # resize to 250x350
    
    rank_img = img[:80,:60,:] 
    rank_img = cv2.cvtColor(cv2.cvtColor(rank_img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    rank_img = np.expand_dims(rank_img, axis=0)
    rank_img = rank_img.astype(np.float32) / 255.0
    
    suit_img = img[80:160,:60,:]
    suit_img = np.expand_dims(suit_img, axis=0)
    suit_img = suit_img.astype(np.float32) / 255.0
    
    return rank_img, suit_img

# ----------------------------------------------------------------------

# Open the default camera (usually camera 0)
#cam = cv2.VideoCapture(2) # This is the index of the Iphone's camera, specific for Member2's Iphone, If you are testing with different Cameras, chaneg the index
cam = cv2.VideoCapture(1)

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
        img = crop_and_warp(frame) # crop and warp
        img = cv2.resize(img, (250, 350)) # resize to 250x350
        
        rank_img, suit_img = load_and_preprocess_frame(frame)

        # Predict
        suit_n = tf.argmax(suit_model.predict(suit_img), axis=1).numpy()[0]
        rank_n = tf.argmax(rank_model.predict(rank_img), axis=1).numpy()[0]
        
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