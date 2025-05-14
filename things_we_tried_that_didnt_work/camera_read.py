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

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['Hearts', 'Diamonds', 'Clubs', 'Spades']

def crop_and_warp(card_contour, original):
    '''crop the background off and warp the card into a perfect rectangle'''
    
    # turn the contour to polygon
    peri = cv2.arcLength(card_contour, True)
    tolerance = 0.02 * peri   # how big of gaps in our contour do we allow (2% of the perimeter)
    approx_poly = cv2.approxPolyDP(card_contour, tolerance, True)

    # if more than 4 corners, something is wrong
    assert len(approx_poly) == 4, "The detected contour has more than 4 corners."
    pts = approx_poly.reshape(4, 2) # reshape to 4 (x,y) pairs

    # get corners
    yx_sum = pts.sum(axis=1)
    yx_diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(yx_sum)] # tl because min x+y
    br = pts[np.argmax(yx_sum)] # br because max x+y
    tr = pts[np.argmin(yx_diff)] # tr because min y-x
    bl = pts[np.argmax(yx_diff)] # bl because max y-x
    warped_rect = np.array([tl, tr, br, bl], dtype="float32")

    # Compute width and height of new image
    # euclidean distance between points
    w = int(np.linalg.norm(br - bl)) 
    h = int(np.linalg.norm(tr - br))

    # Destination points for the target rectangle
    target_rect = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype="float32")

    # Compute the perspective transform matrix and apply it
    transform = cv2.getPerspectiveTransform(warped_rect, target_rect)
    warped = cv2.warpPerspective(original, transform, (w, h))
    #cv2.imwrite("warped.png", warped)

    return warped


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