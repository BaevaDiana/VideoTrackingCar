import cv2

# Load the video file
cap = cv2.VideoCapture('./video_sources/example_3.mp4')

# Create a background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    # Read a frame from the video file
    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction to the frame
    fgmask = fgbg.apply(frame)

    # Find contours in the foreground mask
    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop over the contours
    for contour in contours:
        # Get the bounding box of each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Draw a rectangle around the contour if it's big enough and has a certain aspect ratio
        if w > 50 and h > 50 and w/h > 1.3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Exit if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()
