from imutils import face_utils
import dlib
import cv2

# Initialize dlib's face detector (HOG-based) and create the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# Capture video from the default webcam
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    _, image = cap.read()
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # Detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # Loop over the face detections
    for (i, rect) in enumerate(rects):
        # Predict facial landmarks for the face region
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        # Loop over the facial landmarks and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
        
        # Connect the facial landmarks with lines
        for (start, end) in face_utils.FACIAL_LANDMARKS_IDXS.values():
            pts = shape[start:end + 1]  # Include the last point to close the loop
            cv2.polylines(image, [pts], False, (0, 255, 0), 1)
    
    # Show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    
    # Exit loop when 'Esc' key is pressed
    k = cv2.waitKey(1)
    if k == 27:
        break

# Release video capture object and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()
