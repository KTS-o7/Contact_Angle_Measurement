import cv2
import numpy as np

# Load the image
img = cv2.imread('/home/krishnatejaswis/VSCode/Contact angle/CONTACT ANGLE MEASUREMENT/image.jpg')

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
theta = np.arccos(cos_theta) * (180/np.pi)

# Draw the tangents on the image
rows, cols = img.shape[:2]
x_center, y_center = cols // 2, rows // 2
radius = int(perimeter / (2 * np.pi))
cv2.circle(img, (x_center, y_center), radius, (0, 255, 0), 2)
tan1_x = int(x_center + radius * np.cos(theta/2 * np.pi / 180))
tan1_y = int(y_center - radius * np.sin(theta/2 * np.pi / 180))
tan2_x = int(x_center - radius * np.cos(theta/2 * np.pi / 180))
tan2_y = int(y_center - radius * np.sin(theta/2 * np.pi / 180))
cv2.line(img, (x_center, y_center), (tan1_x, tan1_y), (0, 0, 255), 2)
cv2.line(img, (x_center, y_center), (tan2_x, tan2_y), (0, 0, 255), 2)

# Display the image
cv2.imshow("Contact angle image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()