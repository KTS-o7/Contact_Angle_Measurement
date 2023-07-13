import cv2
import numpy as np

def calculate_contact_angle(img, rod_start, rod_end):
    # Preprocess the image if necessary
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Fit a line to the contour
    [vx, vy, x, y] = cv2.fitLine(largest_contour, cv2.DIST_L2, 0, 0.01, 0.01)
    line_point1 = (int(x - 1000 * vx), int(y - 1000 * vy))
    line_point2 = (int(x + 1000 * vx), int(y + 1000 * vy))

    # Find the intersection point of the rod and liquid-air interface
    intersection_point = get_intersection_point(rod_start, rod_end, line_point1, line_point2)

    # Find the contact angle
    contact_angle = np.arctan2(intersection_point[1] - rod_start[1], intersection_point[0] - rod_start[0])
    contact_angle = np.rad2deg(contact_angle)

    return contact_angle

def get_intersection_point(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator != 0:
        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return int(intersection_x), int(intersection_y)
    else:
        return None

# Video path
video_path = 'rodvideo.mp4'

# Rod coordinates (start and end points)
rod_start = (100, 100)
rod_end = (500, 500)

# Open video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for display
    frame = cv2.resize(frame, (640, 480))

    # Calculate contact angle
    contact_angle = calculate_contact_angle(frame, rod_start, rod_end)

    # Display contact angle on frame
    cv2.putText(frame, f'Contact Angle: {contact_angle:.2f} degrees', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display frame
    cv2.imshow('Video', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
