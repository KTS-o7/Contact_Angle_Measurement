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
# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply a Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection to the image
edges = cv2.Canny(blur, 50, 150)

# Find the contours in the image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
max_area = -1
for contour in contours:
    area = cv2.contourArea(contour)
    if area > max_area:
        max_area = area
        largest_contour = contour

# Fit a polynomial curve to the largest contour
rows, cols = img.shape[:2]
curve_points = np.squeeze(largest_contour)
curve_fit = np.polyfit(curve_points[:,0], curve_points[:,1], 2)
curve_x = np.linspace(0, cols-1, cols)
curve_y = curve_fit[0]*curve_x**2 + curve_fit[1]*curve_x + curve_fit[2]

# Draw the curve on the image
curve_points = np.stack((curve_x, curve_y)).T
curve_points = curve_points.reshape((-1,1,2)).astype(np.int32)
cv2.polylines(img, [curve_points], False, (0, 255, 0), 2)

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

# Draw the tangent line through the selected point
if len(points) == 2:
    # Find the closest point on the curve to the selected point
    selected_point = np.array(points[0])
    closest_point = curve_points[np.argmin(np.linalg.norm(curve_points-selected_point, axis=1))]

    # Find the tangent line to the curve at the closest point
    tangent_angle = np.arctan(2*curve_fit[0]*closest_point[0]+curve_fit[1])
    tangent_direction = np.array([-np.sin(tangent_angle), np.cos(tangent_angle)])
    tangent_line = np.array([closest_point-tangent_direction*100, closest_point+tangent_direction*100]).astype(np.int32)

    # Draw the tangent line on the image
    cv2.line(img, tuple(tangent_line[0]), tuple(tangent_line[1]), (0, 0, 255), 2)

    # Calculate the contact angle
    angle = np.arctan(curve_fit[0]*(2*closest_point[0]-points[1][0])+curve_fit[1]) * 180/np.pi

    # Display the result
    cv2.imshow('Result', img)
    print("Contact angle: ", angle)

cv2.waitKey(0)
cv2.destroyAllWindows()
