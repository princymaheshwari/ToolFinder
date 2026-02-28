import cv2

# Initialize the camera (0 is usually the default camera)
cap = cv2.VideoCapture(4)

if not cap.isOpened():
  print("Error: Could not open video stream.")
  exit()

print("Press 'q' to quit the live feed.")

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  if not ret:
    print("Error: Failed to grab frame.")
    break

  # Display the resulting frame
  cv2.imshow('Live Camera Feed', frame)

  # Break the loop when 'q' is pressed
  if cv2.waitKey(1) == ord('q'):
    break

# Release the capture object and close windows
cap.release()
cv2.destroyAllWindows()
