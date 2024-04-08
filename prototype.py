import cv2
import pyautogui
import numpy as np
import time

# Get the screen size
screen_size = pyautogui.size()

# Define initial region of interest (ROI) around the mouse position
initial_roi = (500, 500, 400, 300)  

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Variables to track the face position and cooldown time
face_detected = False
face_position = (0, 0)
cooldown_time = 10.0  # Adjust this time as needed (in seconds)
last_detection_time = time.time() - cooldown_time  # Initialize last detection time

# Variable to track mouse position for zoom responsiveness
last_mouse_position = pyautogui.position()

# Main loop to continuously capture screen and detect face
while True:
    # Get current mouse position
    mouse_x, mouse_y = pyautogui.position()

    # Reset cooldown time if mouse position changes
    if mouse_x != last_mouse_position[0] or mouse_y != last_mouse_position[1]:
        last_mouse_position = (mouse_x, mouse_y)
        last_detection_time = time.time() - cooldown_time  # Reset cooldown time

    # Calculate ROI position based on mouse position
    roi_x = max(0, mouse_x - initial_roi[2] // 2)
    roi_y = max(0, mouse_y - initial_roi[3] // 2)
    roi_width = initial_roi[2]
    roi_height = initial_roi[3]
    
    # Ensure ROI is within screen bounds
    roi_x = min(max(0, roi_x), screen_size[0] - roi_width)
    roi_y = min(max(0, roi_y), screen_size[1] - roi_height)

    # Define ROI
    roi = (roi_x, roi_y, roi_width, roi_height)
    
    # Capture screenshot of the ROI
    screen_img = pyautogui.screenshot(region=roi)
    
    # Convert screenshot to grayscale
    gray_frame = cv2.cvtColor(np.array(screen_img), cv2.COLOR_RGB2BGR)
    
    # Detect faces only if cooldown time has passed since last detection
    if time.time() - last_detection_time > cooldown_time:
        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Initialize variables for face detection
        face = None
        face_detected = False
        
        # If faces are detected, iterate over each face
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = (x, y, w, h)
                face_detected = True
                break
        # Update last detection time
        last_detection_time = time.time()
    
    # Display the captured screen with or without face detection
    cv2.namedWindow("Screen Capture", cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("Screen Capture", 400, 300)
    if face_detected:
        x, y, w, h = face
        detected_image = gray_frame[y:y + h, x:x + w]
        face_image = cv2.resize(detected_image, (400,300))
    else:
        face_image = gray_frame

    cv2.imshow("Screen Capture", face_image)

    # Calculate the position to display the window at the top middle
    window_width = 400 
    window_height = 300 
    top_middle_x = int((screen_size[0] - window_width) / 2)
    top_middle_y = 10

    # Move the window to the top middle of the screen
    cv2.moveWindow("Screen Capture", top_middle_x, top_middle_y)

    # Check for key press
    key = cv2.waitKey(1)
    if key == ord("q"):  # Quit the loop if 'q' is pressed
        break
    elif key == ord("x"):  # Zoom in if 'x' is pressed
        initial_roi = (roi_x, roi_y, roi_width // 2, roi_height // 2)
        # Adjust the ROI position to keep it centered
        roi_x = max(0, roi_x + roi_width // 4)
        roi_y = max(0, roi_y + roi_height // 4)

    elif key == ord("y"):  # Zoom out if 'y' is pressed
        initial_roi = (roi_x, roi_y, roi_width * 2, roi_height * 2)
        # Adjust the ROI position to keep it centered
        roi_x = max(0, roi_x - roi_width // 2)
        roi_y = max(0, roi_y - roi_height // 2)

# Close all windows when loop ends
cv2.destroyAllWindows()