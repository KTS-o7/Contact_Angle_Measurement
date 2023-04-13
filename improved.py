import cv2
import numpy as np

# Load the image
img = cv2.imread('drop_image.jpg')

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding to isolate the drop
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Find the contours of the drop
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a blank image to draw the contours on
contour_img = np.zeros_like(img)

# Draw the contours on the blank image
cv2.drawContours(contour_img, contours, -1, (255, 255, 255), -1)

# Determine the perimeter of the drop
perimeter = cv2.arcLength(contours[0], True)

# Determine the area of the drop
area = cv2.contourArea(contours[0])

# Calculate the contact angle using the Wilhelm Plate Method
cos_theta = 1 - (2 * area) / (np.pi * (perimeter**2))
