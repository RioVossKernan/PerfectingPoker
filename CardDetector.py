import cv2
import numpy as np
import time
import os
import Cards
import VideoStream
import nbimporter
import tensorflow as tf
from cropped_img_model import CardModel, crop_and_warp
from hand_reader import PokerCard
import matplotlib.pyplot as plt
import hand_reader


## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# ------------------ LOAD VIDEO STREAM -----------------------
vid_src = 1 # 0 for built-in camera, 1 for external camera
videostream = VideoStream.VideoStream(vid_src).start()
time.sleep(1) 

# ------------------ LOAD MODELS -----------------------

rank_model = CardModel(13)
rank_model.build((None, 80, 60, 3))
rank_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
rank_model.load_weights("cropped_rank_model.weights.h5")

suit_model = CardModel(4)
suit_model.build((None, 80, 60, 3))
suit_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
suit_model.load_weights("cropped_suit_model.weights.h5")

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


def crop_and_warp(card_contour, original):
    '''crop the background off and warp the card into a perfect rectangle'''
    
    # turn the contour to polygon
    peri = cv2.arcLength(card_contour, True)
    tolerance = 0.02 * peri   # how big of gaps in our contour do we allow (2% of the perimeter)
    approx_poly = cv2.approxPolyDP(card_contour, tolerance, True)

    # if more than 4 corners, something is wrong
    if len(approx_poly) < 4:
        plt.imshow(original)
        plt.show()
        raise ValueError("The detected contour has less than 4 corners.")
    pts = approx_poly.reshape(4, 2) # reshape to 4 (x,y) pairs

    # get corners
    yx_sum = pts.sum(axis=1)
    yx_diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(yx_sum)] # tl because min x+y
    br = pts[np.argmax(yx_sum)] # br because max x+y
    tr = pts[np.argmin(yx_diff)] # tr because min y-x
    bl = pts[np.argmax(yx_diff)] # bl because max y-x
    center = np.mean(pts, axis=0).astype(int)
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

    return warped, center

def preprocess_frame(contour, frame):
    img, center = crop_and_warp(contour, frame) # crop and warp
    img = cv2.resize(img, (250, 350)) # resize to 250x350
    
    rank = img[:80,:60,:] 
    rank = cv2.cvtColor(cv2.cvtColor(rank, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    suit = img[80:160,:60,:]
    
    return rank, suit, center

ranks = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
suits = ['H', 'D', 'S', 'C']

### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

last_poker_cards = []
last_probs = {}
last_stats = []

# Begin capturing frames
while cam_quit == 0:

    # Grab frame from video stream
    image = videostream.read()

    # Pre-process camera image (gray, blur, and threshold it)
    pre_proc = Cards.preprocess_image(image)
	
    # Find and sort the contours of all cards in the image (query cards)
    cnts_sort, cnt_is_card = Cards.find_cards(pre_proc)

    # If there are no contours, do nothing
    if len(cnts_sort) != 0:

        # Initialize a new "cards" list to assign the card objects.
        # k indexes the newly made array of cards.
        cards = []
        poker_cards = []
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                rank_img, suit_img, center = preprocess_frame(cnts_sort[i], image)
                rank_img = np.expand_dims(rank_img, axis=0)
                suit_img = np.expand_dims(suit_img, axis=0)
                                
                # Find the best rank and suit match for the card.
                suit_n = tf.argmax(suit_model.predict(suit_img, verbose=0), axis=1).numpy()[0]
                rank_n = tf.argmax(rank_model.predict(rank_img, verbose=0), axis=1).numpy()[0]

                suit_name = suits[suit_n]
                rank_name = ranks[rank_n]
                
                # Poker card is used for simulations
                poker_card = PokerCard(rank_name, suit_name)
                poker_cards.append(poker_card)
                
                # Query card is used for drawinf
                query_card = Cards.Query_card()
                query_card.best_rank_match = rank_name
                query_card.best_suit_match = suit_name
                query_card.center = center
                cards.append(query_card)

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1
        
        if 1 <= len(poker_cards) <= 5:
            last_poker_cards = poker_cards
            last_probs = hand_reader.estimate_hand_probabilities(poker_cards, simulations=200)
        else:
            last_poker_cards = poker_cards
            last_probs = {}
        card_text = f"Detected Cards: {' '.join(str(card) for card in last_poker_cards)}"
        cv2.putText(image, card_text, (10, 50), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        if last_probs:
            y_offset = 100  
            cv2.putText(image, "Probabilities:", (10, y_offset), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 30  
            
            for hand, prob in sorted(last_probs.items()):
                text = f"{hand}: {prob:.2%}"
                cv2.putText(image, text, (30, y_offset), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                y_offset += 30


    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector",image)

    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        

# Close all windows and close the video stream.
cv2.destroyAllWindows()
videostream.stop()