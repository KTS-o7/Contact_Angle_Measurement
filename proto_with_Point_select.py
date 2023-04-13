import cv2
import numpy as np

# Create a variable to store the selected points
points = []

# Mouse callback function to store the selected points
def select_point(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONUP:
        points.append((x, y))

# Load the input image
img = cv2.imread('/home/krishnatejaswis/VSCode/Contact angle/CONTACT ANGLE MEASUREMENT/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to the image
edges = cv2.Canny(blur, 50, 150)

# Find the contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Fit a straight line to the largest contour
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)

# Draw the line on the image
cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)

# Set the mouse callback function
cv2.namedWindow('Result')
cv2.setMouseCallback('Result', select_point)

# Wait for points to be selected
while True:
    cv2.imshow('Result', img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif len(points) == 2:
        break

# Draw the line through the selected points
if len(points) == 2:
    cv2.line(img, points[0], points[1], (0, 0, 255), 2)

    # Calculate the contact angle
    angle = np.arctan((points[1][1]-points[0][1])/(points[1][0]-points[0][0])) * 180/np.pi

    # Display the result
    cv2.imshow('Result', img)
    print("Contact angle: ", angle)

cv2.waitKey(0)
cv2.destroyAllWindows()
