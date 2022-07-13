"""
https://www.youtube.com/watch?v=DFFQM6Zm-8c
"""

import numpy as np
import cv2
import imutils
import scipy.spatial.distance as dist
import matplotlib.pyplot as plt
import pandas as pd

def image_segmentation():
    cv2.destroyAllWindows()

    img = cv2.imread('shapes.png')   # Read the image
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)    # create HSV image
    
    # Define HSV range for yellow color:
    low_yellow = np.array([20, 100, 100])
    high_yellow = np.array([30, 255, 255])
    
    mask1 = cv2.inRange(hsv, low_yellow, high_yellow)
    img1 = cv2.bitwise_and(hsv, hsv, mask = mask1)
    
    # Define HSV range for red color:
    low_red = np.array([161, 155, 84])
    high_red = np.array([179, 255, 255])
    
    mask_2 = cv2.inRange(hsv, low_red, high_red)
    img2 = cv2.bitwise_and(hsv, hsv, mask = mask_2)
    
    # Define HSV range for blue color:
    low_blue = np.array([94, 80, 2])
    high_blue = np.array([126, 255, 255])
    
    mask_3 = cv2.inRange(hsv, low_blue, high_blue)
    img3 = cv2.bitwise_and(hsv, hsv, mask = mask_3)
    
    return img1, img2, img3

def midPoint(ptA, ptB):
    return((ptA[0]+ptB[0])/2, (ptA[1]+ptB[1])/2)

def measure_(image_name):

    image_ = image_name
    
    gray = cv2.cvtColor(image_, cv2.COLOR_BGR2GRAY)
    
    # Remove noises
    tresh, tresh_img = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
    
    # find the total contours
    conts = cv2.findContours(tresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    # This is the total contours
    conts = imutils.grab_contours(conts)
    
    # Create empty image with the original dimension
    cont_img = np.zeros(image_.shape)
    
    # Draw the contours in the empty image
    cont_img = cv2.drawContours(cont_img, conts, -1, (0,255,0), 2)
    
    # Calculate and print all the dimension of individual object    
    
    dim_A = []
    dim_B = []
    
    for c in conts:
        # Extract box points
        box = cv2.minAreaRect(c)
        box = cv2.boxPoints(box)
        
        # Convert the box points into integer
        box = np.array(box, dtype = int)
        
        # Draw the contour in the loop iterations
        cv2.drawContours(cont_img, [c], -1, (2,255,0), 1)

        # Draw bonding box for each individual box
        cv2.drawContours(cont_img, [box], -1, (255,255,255), 1)
        
        for (x,y) in box:
            cv2.circle(cont_img, (x, y), 2, (255,0,0), 2)
            (tl, tr, br, bl) = box
            (tlX, trX) = midPoint(tl, tr)
            (brX, blX) = midPoint(bl, br)
            
            cv2.circle(cont_img, (int(tlX), int(trX)), 1, (255,0,0), 2)    # midpoint at top side
            cv2.circle(cont_img, (int(brX), int(blX)), 1, (255,0,0), 2)    # midpoint at bottom side
            
            cv2.line(cont_img, (int(tlX), int(trX)), (int(brX), int(blX)), (255,255,255), 1)  # draw line connecting midpoint at top side and midpoint at bottom side
            
            dA = dist.euclidean((tlX, trX),(brX, blX))   # Distance between 2 points (midpoint at top side and midpoint at bottom side)
            
            # print the size in pixel on each contour
            cv2.putText(cont_img, "{:.1f} px".format(dA), (int(tlX)-10, int(trX)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            
            (tlX, trX) = midPoint(tl, bl)
            (brX, blX) = midPoint(tr, br)
            
            cv2.circle(cont_img, (int(tlX), int(trX)), 1, (255,0,0), 2)    # midpoint at left side
            cv2.circle(cont_img, (int(brX), int(blX)), 1, (255,0,0), 2)  # midpoint at right side
            
            cv2.line(cont_img, (int(tlX), int(trX)), (int(brX), int(blX)), (255,255,255), 1)  # draw line connecting midpoint at left side and midpoint at right side
            
            dB = dist.euclidean((tlX, trX),(brX, blX))   # Distance between 2 points (midpoint at top side and midpoint at bottom side)
            
            # print the size in pixel on each contour
            cv2.putText(cont_img, "{:.1f} px".format(dB), (int(brX)+10, int(blX)+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        
        dim_A.append(dA)
        dim_B.append(dB)
    
    plt.imshow(cont_img)
    dim_A = np.array(dim_A)
    dim_B = np.array(dim_B)
    dim_A = dim_A.reshape(-1, 1)
    dim_B = dim_B.reshape(-1, 1)
    dim_A_B = np.hstack((dim_A,dim_B))
    dim_A_B = pd.DataFrame(dim_A_B, columns=['dA','dB'])
    
    return dim_A_B

img_yellow, img_red, img_blue = image_segmentation()

dim_A_B = measure_(img_yellow)

