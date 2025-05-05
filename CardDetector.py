import cv2
import numpy as np
import time
import os
import Cards
import VideoStream

import hand_reader
### ---- INITIALIZATION ---- ###
# Define constants and initialize variables

## Camera settings
IM_WIDTH = 1280
IM_HEIGHT = 720
FRAME_RATE = 10


## Define font to use
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera object and video feed from the camera. The video stream is set up
# as a seperate thread that constantly grabs frames from the camera feed. 
# See VideoStream.py for VideoStream class definition
videostream = VideoStream.VideoStream((IM_WIDTH, IM_HEIGHT), FRAME_RATE, 1).start()
time.sleep(1) # Give the camera time to warm up

# Load the train rank and suit images
path = os.path.dirname(os.path.abspath(__file__))
train_ranks = Cards.load_ranks( path + '/Card_Imgs/')
train_suits = Cards.load_suits( path + '/Card_Imgs/')


### ---- MAIN LOOP ---- ###
# The main loop repeatedly grabs frames from the video stream
# and processes them to find and identify playing cards.

cam_quit = 0 # Loop control variable

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
        k = 0

        # For each contour detected:
        for i in range(len(cnts_sort)):
            if (cnt_is_card[i] == 1):

                # Create a card object from the contour and append it to the list of cards.
                # preprocess_card function takes the card contour and contour and
                # determines the cards properties (corner points, etc). It generates a
                # flattened 200x300 image of the card, and isolates the card's
                # suit and rank from the image.
                cards.append(Cards.preprocess_card(cnts_sort[i],image))

                # Find the best rank and suit match for the card.
                cards[k].best_rank_match,cards[k].best_suit_match,cards[k].rank_diff,cards[k].suit_diff = Cards.match_card(cards[k],train_ranks,train_suits)
                poker_cards = hand_reader.convert_detected_cards_to_poker_cards(cards)
    
                if poker_cards:
                    last_poker_cards = poker_cards
                    last_stats = hand_reader.get_hand_stats(poker_cards)
                    if 1 <= len(poker_cards) <= 5:
                        last_probs = hand_reader.estimate_hand_probabilities(poker_cards, simulations=200)
                
                card_text = f"Detected Cards: {' '.join(str(card) for card in last_poker_cards)}"
                cv2.putText(image, card_text, (10, 50), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                if last_probs:
                    prob_text = f"Probabilities: {', '.join(f'{hand}: {prob:.2%}' for hand, prob in last_probs.items())}"
                    cv2.putText(image, prob_text, (10, 100), font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            

                # Draw center point and match result on the image.
                image = Cards.draw_results(image, cards[k])
                k = k + 1
	    
        # Draw card contours on image (have to do contours all at once or
        # they do not show up properly for some reason)
        if (len(cards) != 0):
            temp_cnts = []
            for i in range(len(cards)):
                temp_cnts.append(cards[i].contour)
            cv2.drawContours(image,temp_cnts, -1, (255,0,0), 2)
        

    # Finally, display the image with the identified cards!
    cv2.imshow("Card Detector",image)

    
    # Poll the keyboard. If 'q' is pressed, exit the main loop.
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        cam_quit = 1
        

# Close all windows and close the video stream.
cv2.destroyAllWindows()
videostream.stop()