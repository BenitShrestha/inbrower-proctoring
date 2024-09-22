import cv2
from deepface import DeepFace

# Load the reference image
reference_image_path = 'Simple_Live_Face_Recognition/Images/manav.png'
reference_image = cv2.imread(reference_image_path)

# Ensure the reference image is loaded properly
assert reference_image is not None, "Reference image could not be loaded"

# Initialize the webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Change to 1 if using an external camera

# Set the resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Frame processing parameters
frame_skip = 10  # Number of frames to skip
frame_count = 0

# Main loop for face detection and comparison
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to capture frame")
        break

    # Skip frames
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue  # Skip the rest of the loop for this frame

    # Resize the frame to speed up processing
    frame_resized = cv2.resize(frame, (320, 240))  # Resize to a smaller resolution

    # Use DeepFace to verify the captured frame against the reference image
    try:
        result = DeepFace.verify(frame_resized, reference_image, model_name='Facenet512', detector_backend='opencv')
        if result['verified']:
            message = "Face Matched!"
            color = (0, 255, 0)  # Green for a match
        else:
            message = "Not Recognized!"
            color = (0, 0, 255)  # Red for no match

    except Exception as e:
        print(f"Face verification failed: {e}")
        message = "Error!"
        color = (255, 0, 0)  # Blue for errors

    # Display the result on the frame
    cv2.putText(frame, message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    # Show the frame with the result
    cv2.imshow('Face Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()
