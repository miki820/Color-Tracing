import cv2
import numpy as np

# Open Image
image = cv2.imread("files/ball.png")

# Original Image
cv2.imshow("Ball", image)
cv2.waitKey(0)

# Convert Image to HSV Colors
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imshow("BallHSV", hsv_img)
cv2.waitKey(0)

# Define lower and upper bounds for the color of the ball in HSV
lower = np.array([0, 100, 100])
upper = np.array([25, 255, 255])

# Creating a mask
mask_img = cv2.inRange(hsv_img, lower, upper)
cv2.imshow("Ball Mask", mask_img)
cv2.waitKey(0)

# Kernel definition for morphological operations
kernel = np.ones((5, 5), np.uint8)

# Removing noise from the mask using morphological opening
mask_img_without_noise = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel)
cv2.imshow("Ball Mask without noise", mask_img_without_noise)
cv2.waitKey(0)

# Finding contours in the mask
contours, hierarchy = cv2.findContours(mask_img_without_noise, 1, 2)

# Finding largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Calculate centroid of the largest contour
M = cv2.moments(largest_contour)
cx = int(M['m10'] / M['m00'])
cy = int(M['m01'] / M['m00'])

# Draw a marker at the centroid on the original image
image_marker = image.copy()
cv2.drawMarker(image_marker, (int(cx), int(cy)), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
cv2.imshow('Ball Middle', image_marker)
cv2.waitKey(0)

cv2.destroyAllWindows()