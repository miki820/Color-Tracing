import cv2
import numpy as np

# Open the video
video = cv2.VideoCapture()
video.open('files/movingball.mp4')

# Get total number of frames and frame size
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(video.get(3))
frame_height = int(video.get(4))
size = (frame_width, frame_height)

# Create VideoWriter object to save processed video
result = cv2.VideoWriter(
    'files/result.avi',  cv2.VideoWriter_fourcc(*'MJPG'), 20, size)

# Define lower and upper bounds for the color of the ball in HSV
lower = np.array([0, 100, 100])
upper = np.array([25, 255, 255])

# Process each frame in the video
for counter in range(total_frames):

    # Read frame
    success, frame_rgb = video.read()
    if not success:
        break

    # Calculating progress
    progress = int((counter / total_frames) * 100)
    print('\rGenerowanie Filmiku: {}%'.format(progress), end='')

    # Convert the frame to HSV
    hsv_frame = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2HSV)

    # Create a mask to isolate a ball
    mask = cv2.inRange(hsv_frame, lower, upper)

    # Finding contours
    contours, _ = cv2.findContours(mask, 1, 2)
    largest_contour = max(contours, key=cv2.contourArea)

    # Calculate centroid
    M = cv2.moments(largest_contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # Drawing a markerat the centroid
        cv2.drawMarker(frame_rgb, (int(cx), int(cy)), color=(
            0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=3)

    # Write a frame with the marker to the result video
    result.write(frame_rgb)

# Release video objects
video.release()
result.release()
