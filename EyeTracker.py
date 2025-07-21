import cv2
import mediapipe as mp
import numpy as np
import math
import time
#from sklearn.linear_model import LinearRegression # For calibration mapping
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV # 
from sklearn.preprocessing import StandardScaler # <-- Good practice for SVMs
import matplotlib.pyplot as plt # 
import joblib # 
import os 

# Initialize MediaPipe Face Mesh and Drawing Utilities
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Global Parameters ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# Your monitor's resolution (in pixels)
SCREEN_WIDTH_PX = 1920
SCREEN_HEIGHT_PX = 1080
# Physical dimensions of your screen (e.g., a 24-inch monitor might be ~531mm wide, 298mm high)
# These are used to model the screen in 3D space. MEASURE YOUR OWN SCREEN!
SCREEN_WIDTH_MM = 385
SCREEN_HEIGHT_MM = 216

# Camera's position relative to the CENTER of the screen in millimeters.
# IMPORTANT: Adjust these based on your setup.
# X: Positive to the right of screen center, Negative to the left.
# Y: Positive upwards from screen center, Negative downwards.
# Z: Positive towards the user (closer to screen), Negative away from the user.
# Example: Camera 500mm from screen, centered horizontally, slightly above center.
SCREEN_DISTANCE_MM = 510 # Distance from camera (lens) to screen center
CAMERA_OFFSET_X_MM = 420   # How far left/right the camera is from screen center line
CAMERA_OFFSET_Y_MM = 280 # How far up/down the camera is from screen center line (negative for above center)
CAMERA_OFFSET_Z_MM = 50   # Usually 0, or positive if camera is behind the screen plane (unlikely)

# Calibration points (9 points: corners, midpoints, center) normalized to 0-1 range
# These will be scaled to SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX
CALIBRATION_POINTS = [
    (0.5, 0.5), # Center
    (0.05, 0.05), # Top-Left
    (0.33, 0.05), # Top-center-left
    (0.66, 0.05), # Top-center-Right
    (0.95, 0.05), # Top-Right
    (0.05, 0.95), # Bottom-Left
    (0.33, 0.95), # Botton-center-left
    (0.66, 0.95), # Botton-center-Right
    (0.95, 0.95), # Bottom-Right
    (0.5, 0.05), # Top-Middle
    (0.5, 0.33), # Top-center-Middle
    (0.5, 0.66), # Bottom-center-Middle
    (0.5, 0.95), # Bottom-Middle
    (0.05, 0.5), # Left-Middle
    (0.33, 0.5), # Top-center-Middle
    (0.66, 0.5), # Bottom-center-Middle
    (0.95, 0.5)  # Right-Middle
]
CALIBRATION_FRAME_COUNT = 100 # Number of frames to collect data for each point

# Calibration models (will be populated after calibration)
gaze_model_x = None
gaze_model_y = None
scaler = None

def get_3d_head_pose(face_landmarks, image_width, image_height, camera_matrix, dist_coeffs):
    """
    Estimates the 3D head pose (rotation and translation) using MediaPipe landmarks.

    Args:
        face_landmarks: MediaPipe FaceMesh.FaceLandmarks object.
        image_width: Width of the image frame.
        image_height: Height of the image frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.

    Returns:
        tuple: (rotation_vector, translation_vector) or (None, None) if detection fails.
    """
    # Define the 3D model points of a generic human face
    # These are approximate values based on a standard face model (e.g., from dlib or similar).
    # MediaPipe landmarks used (indices from Face Mesh):
    # Nose Tip: 1
    # Chin: 199
    # Left Eye Left Corner: 33
    # Right Eye Right Corner: 263
    # Left Mouth Corner: 61
    # Right Mouth Corner: 291
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip (index 1) - reference point
        (0.0, -330.0, -65.0),        # Chin (index 199)
        (-225.0, 170.0, -135.0),     # Left Eye Left Corner (index 33)
        (225.0, 170.0, -135.0),      # Right Eye Right Corner (index 263)
        (-150.0, -150.0, -125.0),    # Left Mouth Corner (index 61)
        (150.0, -150.0, -125.0)      # Right Mouth Corner (index 291)
    ], dtype=np.float64)

    # Convert MediaPipe normalized landmark coordinates to pixel coordinates
    image_points = []
    
    # Specific landmarks to use for pose estimation
    selected_landmark_indices = [1, 199, 33, 263, 61, 291] 

    # Robustly collect image points
    for idx in selected_landmark_indices:
        if idx < len(face_landmarks.landmark): # Check if landmark index exists
            lm = face_landmarks.landmark[idx]
            x = lm.x * image_width
            y = lm.y * image_height
            image_points.append((x, y))
        else:
            # If a critical landmark is missing, we cannot proceed with pose estimation
            return None, None 
    
    if len(image_points) != len(model_points):
        # This implies not all required landmarks were found
        return None, None

    image_points = np.array(image_points, dtype=np.float64)

    # SolvePnP: Finds the object pose from 2D-3D point correspondences
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        return rotation_vector, translation_vector
    else:
        return None, None

def draw_head_pose_axes(image, rotation_vector, translation_vector, camera_matrix, dist_coeffs):
    """
    Draws 3D axes (X, Y, Z) on the head to visualize its pose.
    """
    if rotation_vector is None or translation_vector is None:
        return image

    # Define axis points (length 100 units) in the model coordinate system
    axis = np.float32([[100, 0, 0], [0, 100, 0], [0, 0, 100], [0, 0, 0]]).reshape(-1, 3)

    # Project these 3D points to the 2D image plane
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    # Draw the axes: X (Red), Y (Green), Z (Blue)
    image = cv2.line(image, tuple(imgpts[3].ravel().astype(int)), tuple(imgpts[0].ravel().astype(int)), (0, 0, 255), 3)
    image = cv2.line(image, tuple(imgpts[3].ravel().astype(int)), tuple(imgpts[1].ravel().astype(int)), (0, 255, 0), 3)
    image = cv2.line(image, tuple(imgpts[3].ravel().astype(int)), tuple(imgpts[2].ravel().astype(int)), (255, 0, 0), 3)
    
    return image

def get_gaze_direction_from_3d_pupils(face_landmarks, image_width, image_height, camera_matrix, dist_coeffs, rotation_vector, translation_vector):
    """
    Estimates gaze direction (Left/Right/Center/Blinking) using 3D pupil coordinates
    and head pose. This is still a heuristic but leverages 3D information.

    Args:
        face_landmarks: MediaPipe FaceMesh.FaceLandmarks object.
        image_width: Width of the frame.
        image_height: Height of the frame.
        camera_matrix: Camera intrinsic matrix.
        dist_coeffs: Camera distortion coefficients.
        rotation_vector: 3D head rotation vector from solvePnP.
        translation_vector: 3D head translation vector from solvePnP.

    Returns:
        tuple: (gaze_text_string, left_pupil_2d, right_pupil_2d)
    """
    if rotation_vector is None or translation_vector is None:
        return "No Head Pose", (None, None), (None, None)

    # Indices for the iris landmarks (available when refine_landmarks=True)
    PUPIL_LEFT_MP = 468
    PUPIL_RIGHT_MP = 473

    # Indices for eye corners for blink detection and horizontal range
    LEFT_EYE_TOP = 159 
    LEFT_EYE_BOTTOM = 145 
    LEFT_EYE_LEFT_CORNER = 33
    LEFT_EYE_RIGHT_CORNER = 133

    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    RIGHT_EYE_LEFT_CORNER = 362
    RIGHT_EYE_RIGHT_CORNER = 263

    # Helper function to get 2D pixel coordinates from MediaPipe normalized landmarks, with robustness
    def get_2d_coords(idx):
        if idx >= len(face_landmarks.landmark): # Check if landmark index exists
            return None # Return None if index is out of bounds
        lm = face_landmarks.landmark[idx]
        return int(lm.x * image_width), int(lm.y * image_height)

    # Get 2D pupil coordinates for drawing and calculations
    pupil_left_2d = get_2d_coords(PUPIL_LEFT_MP)
    pupil_right_2d = get_2d_coords(PUPIL_RIGHT_MP)

    # Check if pupils are detected. If not, return early.
    if pupil_left_2d is None or pupil_right_2d is None:
        return "Pupils Not Detected", (None, None), (None, None)

    # Get eye vertical aperture for blink detection
    left_eye_top = get_2d_coords(LEFT_EYE_TOP)
    left_eye_bottom = get_2d_coords(LEFT_EYE_BOTTOM)
    right_eye_top = get_2d_coords(RIGHT_EYE_TOP)
    right_eye_bottom = get_2d_coords(RIGHT_EYE_BOTTOM)

    # Check if all eye boundary points are found for blink detection. If not, return early.
    if (left_eye_top is None or left_eye_bottom is None or
        right_eye_top is None or right_eye_bottom is None):
        return "Eye Boundaries Missing", pupil_left_2d, pupil_right_2d


    left_eye_v_dist = math.hypot(left_eye_top[0] - left_eye_bottom[0], left_eye_top[1] - left_eye_bottom[1])
    right_eye_v_dist = math.hypot(right_eye_top[0] - right_eye_bottom[0], right_eye_top[1] - right_eye_bottom[1])

    BLINK_THRESHOLD = 10 # Pixels, adjust based on camera/resolution

    if left_eye_v_dist < BLINK_THRESHOLD or right_eye_v_dist < BLINK_THRESHOLD:
        return "Blinking", pupil_left_2d, pupil_right_2d

    # Use the 2D pupil positions relative to the eye corners for horizontal gaze
    le_left_2d = get_2d_coords(LEFT_EYE_LEFT_CORNER)
    le_right_2d = get_2d_coords(LEFT_EYE_RIGHT_CORNER)
    re_left_2d = get_2d_coords(RIGHT_EYE_LEFT_CORNER)
    re_right_2d = get_2d_coords(RIGHT_EYE_RIGHT_CORNER)

    # Check if eye corner points are found for horizontal gaze. If not, return early.
    if (le_left_2d is None or le_right_2d is None or
        re_left_2d is None or re_right_2d is None):
        return "Eye Corners Missing", pupil_left_2d, pupil_right_2d


    # Calculate horizontal center of the eye in 2D
    left_eye_center_x_2d = (le_left_2d[0] + le_right_2d[0]) // 2
    right_eye_center_x_2d = (re_left_2d[0] + re_right_2d[0]) // 2

    # Gaze offset based on pupil position relative to eye center
    left_gaze_offset = pupil_left_2d[0] - left_eye_center_x_2d
    right_gaze_offset = pupil_right_2d[0] - right_eye_center_x_2d

    # Adjust thresholds based on the accuracy of your setup
    LOOK_LEFT_THRESHOLD = -5
    LOOK_RIGHT_THRESHOLD = 5

    if left_gaze_offset < LOOK_LEFT_THRESHOLD and right_gaze_offset < LOOK_LEFT_THRESHOLD:
        return "Looking Left", pupil_left_2d, pupil_right_2d
    elif left_gaze_offset > LOOK_RIGHT_THRESHOLD and right_gaze_offset > LOOK_RIGHT_THRESHOLD:
        return "Looking Right", pupil_left_2d, pupil_right_2d
    else:
        return "Looking Center", pupil_left_2d, pupil_right_2d

###############

# --- NEW: Feature Extraction for Calibration/Prediction ---
def extract_gaze_features(face_landmarks, image_width, image_height, rotation_vector, translation_vector):
    """
    Extracts a comprehensive set of features from face_landmarks and head pose.
    Includes individual pupil offsets, normalized eye measurements, head pose angles,
    and the user's distance from the camera (tvec_z).

    Args:
        face_landmarks: MediaPipe FaceMesh.FaceLandmarks object.
        image_width: Width of the image frame.
        image_height: Height of the image frame.
        rotation_vector: 3D head rotation vector from solvePnP.
        translation_vector: 3D head translation vector from solvePnP.

    Returns:
        list: A feature vector, or None if critical landmarks are missing.
    """
    PUPIL_LEFT_MP = 468
    PUPIL_RIGHT_MP = 473
    LEFT_EYE_LEFT_CORNER = 33
    LEFT_EYE_RIGHT_CORNER = 133
    RIGHT_EYE_LEFT_CORNER = 362
    RIGHT_EYE_RIGHT_CORNER = 263
    
    LEFT_EYE_TOP = 159 
    LEFT_EYE_BOTTOM = 145     
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
   
    def get_2d_coords(idx):
        if idx >= len(face_landmarks.landmark):
            return None
        lm = face_landmarks.landmark[idx]
        return int(lm.x * image_width), int(lm.y * image_height)

    pupil_left_2d = get_2d_coords(PUPIL_LEFT_MP)
    pupil_right_2d = get_2d_coords(PUPIL_RIGHT_MP)
    
    le_left_2d = get_2d_coords(LEFT_EYE_LEFT_CORNER)
    le_right_2d = get_2d_coords(LEFT_EYE_RIGHT_CORNER)
    re_left_2d = get_2d_coords(RIGHT_EYE_LEFT_CORNER)
    re_right_2d = get_2d_coords(RIGHT_EYE_RIGHT_CORNER)

    left_eye_top = get_2d_coords(LEFT_EYE_TOP)
    left_eye_bottom = get_2d_coords(LEFT_EYE_BOTTOM)
    right_eye_top = get_2d_coords(RIGHT_EYE_TOP)
    right_eye_bottom = get_2d_coords(RIGHT_EYE_BOTTOM)
    
    # Ensure all required landmarks are available for feature extraction
    required_landmarks = [
        pupil_left_2d, pupil_right_2d,
        le_left_2d, le_right_2d, re_left_2d, re_right_2d,
        left_eye_top, left_eye_bottom, right_eye_top, right_eye_bottom
    ]
    
    if any(p is None for p in required_landmarks):
        print("Warning: Missing critical eye landmarks for feature extraction.")
        return None

    # Pupil offsets relative to eye center
    left_eye_center_x_2d = (le_left_2d[0] + le_right_2d[0]) // 2
    left_eye_center_y_2d = (left_eye_top[1] + left_eye_bottom[1]) // 2 # Use top/bottom for vertical center

    right_eye_center_x_2d = (re_left_2d[0] + re_right_2d[0]) // 2
    right_eye_center_y_2d = (right_eye_top[1] + right_eye_bottom[1]) // 2
    #Raw pupil offset
    left_pupil_offset_x = pupil_left_2d[0] - left_eye_center_x_2d
    left_pupil_offset_y = pupil_left_2d[1] - left_eye_center_y_2d
    right_pupil_offset_x = pupil_right_2d[0] - right_eye_center_x_2d
    right_pupil_offset_y = pupil_right_2d[1] - right_eye_center_y_2d
    
    avg_pupil_offset_x = ( (left_pupil_offset_x) + 
                           (right_pupil_offset_x) ) / 2
    avg_pupil_offset_y = ( (left_pupil_offset_y) +
                           (right_pupil_offset_y) ) / 2
    # Eye Widths (horizontal span)
    left_eye_width = math.hypot(le_right_2d[0] - le_left_2d[0], le_right_2d[1] - le_left_2d[1])
    right_eye_width = math.hypot(re_right_2d[0] - re_left_2d[0], re_right_2d[1] - re_left_2d[1])


    # Normalized Pupil Offsets (relative to eye width/height to make them scale-invariant)
    # Avoid division by zero
    norm_left_pupil_offset_x = left_pupil_offset_x / left_eye_width if left_eye_width != 0 else 0
    norm_right_pupil_offset_x = right_pupil_offset_x / right_eye_width if right_eye_width != 0 else 0
    # Eye Aspect Ratio (EAR) - a common feature for vertical eye state (blink, open, squint)
    # Using eye vertical distances and horizontal distances for normalization
    left_eye_vertical_dist = math.hypot(left_eye_top[0] - left_eye_bottom[0], left_eye_top[1] - left_eye_bottom[1])
    right_eye_vertical_dist = math.hypot(right_eye_top[0] - right_eye_bottom[0], right_eye_top[1] - right_eye_bottom[1])

    # Normalize vertical offset by eye height (EAR is better for general vertical state)
    norm_left_pupil_offset_y = left_pupil_offset_y / left_eye_vertical_dist if left_eye_vertical_dist != 0 else 0
    norm_right_pupil_offset_y = right_pupil_offset_y / right_eye_vertical_dist if right_eye_vertical_dist != 0 else 0

    # --- Head Pose Features (Yaw, Pitch, Roll) ---
    R, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6 # Check for gimbal lock

    if not singular:
        pitch_rad = math.atan2(R[2,1] , R[2,2])
        yaw_rad = math.atan2(-R[2,0], sy)
        roll_rad = math.atan2(R[1,0], R[0,0])
    else: # Handle gimbal lock case (though rare for typical head movements)
        pitch_rad = math.atan2(-R[1,2], R[1,1])
        yaw_rad = math.atan2(-R[2,0], sy)
        roll_rad = 0

    pitch_deg = math.degrees(pitch_rad)
    yaw_deg = math.degrees(yaw_rad)
    roll_deg = math.degrees(roll_rad)    
    
    # --- Head Translation Features (Distance to Camera) ---
    # Translation vector gives (X, Y, Z) position of the face's origin point (nose tip in model_points)
    # relative to the camera in its own coordinate system.
    # tvec_z is particularly important as it represents distance from camera.
    tvec_x = translation_vector[0][0]
    tvec_y = translation_vector[1][0]
    tvec_z = translation_vector[2][0]

    # --- Combine all features into a single vector ---
    # Experiment with which features work best.
    # A richer set gives the model more information, but too many can lead to overfitting or noise if not relevant.
    feature_vector = [
        norm_left_pupil_offset_x,
        norm_left_pupil_offset_y,
        norm_right_pupil_offset_x,
        norm_right_pupil_offset_y,
        left_eye_width, # The actual width in pixels, might indicate distance or face size
        right_eye_width,
        left_eye_vertical_dist, # Actual height in pixels, used for blink and vertical gaze context
        right_eye_vertical_dist,
        yaw_deg,
        pitch_deg,
        roll_deg,
        tvec_x, # Horizontal position relative to camera
        tvec_y, # Vertical position relative to camera
        tvec_z  # Distance from camera (crucial for perspective)
    ]

    return feature_vector


# --- NEW: Screen Gaze Estimation using Calibration Models ---
def estimate_screen_gaze_calibrated(features, screen_width_px, screen_height_px):
    """
    Predicts screen coordinates using the trained regression models.
    """
    global gaze_model_x, gaze_model_y

    if gaze_model_x is None or gaze_model_y is None:
        return None, None # Models not trained yet

    if features is None:
        return None, None

    features_np = np.array(features).reshape(1, -1) # Reshape for single prediction

    screen_x = gaze_model_x.predict(features_np)[0]
    screen_y = gaze_model_y.predict(features_np)[0]

    # Clamp coordinates to screen boundaries
    screen_x = int(max(0, min(screen_x, screen_width_px - 1)))
    screen_y = int(max(0, min(screen_y, screen_height_px - 1)))

    return screen_x, screen_y


######################---MAIN--FUNCTION---

def mediapipe_3d_pose_and_gaze():
    """
    Performs real-time 3D head pose estimation, enhanced gaze direction,
    and estimates screen focus coordinates using MediaPipe Face Mesh.
    Includes a calibration phase.
    """
    # Initialize Face Mesh with refine_landmarks=True to get iris landmarks
    # max_num_faces=1: focuses on a single face for performance and simplicity
    # min_detection_confidence=0.5: requires 50% confidence to detect a face
    # min_tracking_confidence=0.5: requires 50% confidence to track landmarks after initial detection
    
    global gaze_model_x, gaze_model_y, scaler
    
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True, # THIS IS CRUCIAL for getting iris landmarks (indices 468+)
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0) # Open the default webcam (index 0 for main camera)

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure it's not in use by another application and is properly connected. Exiting.")
        return

    # --- Set a fixed resolution for the webcam for consistency ---
    # This often helps with MediaPipe's internal graph stability and predictable behavior.
    desired_width = 640
    desired_height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

    # Read the actual dimensions set by the camera.
    # Note: The camera might not support the desired resolution exactly,
    # so we read back what it actually set.
    image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Webcam initialized at resolution: {image_width}x{image_height}")

    # --- Camera Intrinsic Parameters (Refined Placeholder) ---
    # These are crucial for accurate 3D pose estimation.
    # For robust production use, perform a dedicated camera calibration.
    # fx, fy: focal lengths in pixels (approximated as image width for simple webcams)
    # cx, cy: principal point (optical center, usually the center of the image)
    focal_length_x = image_width # Approximating focal length with image width
    focal_length_y = image_width # Assuming square pixels, so fx ~ fy
    center_x = image_width / 2
    center_y = image_height / 2
    
    camera_matrix = np.array([
        [focal_length_x, 0, center_x],
        [0, focal_length_y, center_y],
        [0, 0, 1]], dtype="double")

    # Assuming no lens distortion. For real-world accuracy, obtain distortion coefficients via calibration.
    dist_coeffs = np.zeros((4, 1)) # All zeros means no distortion

    print("\n--- Starting Gaze Tracker ---")
    print(f"Assuming screen resolution: {SCREEN_WIDTH_PX}x{SCREEN_HEIGHT_PX}")
    print(f"Camera offset from screen center: ({CAMERA_OFFSET_X_MM}, {CAMERA_OFFSET_Y_MM}, {CAMERA_OFFSET_Z_MM}) mm")
    
    #--TO LOAD PREVIOUS MODEL--
    load_model = input("\nDo you want to load a previously trained model? (y/N): ").strip().lower()
    if load_model == 'y':
        model_path = input("Enter the path to the stored models (e.g., models/my_gaze_model.joblib): ").strip()
        if os.path.exists(model_path):
            try:
                loaded_data = joblib.load(model_path)
                gaze_model_x = loaded_data['model_x']
                gaze_model_y = loaded_data['model_y']
                scaler = loaded_data['scaler'] # Load the scaler too!
                print(f"Models and scaler loaded successfully from {model_path}")
                # Skip calibration if model is loaded
                skip_calibration = True
            except Exception as e:
                print(f"Error loading models: {e}. Proceeding with calibration.")
                skip_calibration = False
        else:
            print(f"Model file not found at {model_path}. Proceeding with calibration.")
            skip_calibration = False
    else:
        skip_calibration = False
    if not skip_calibration:
        print("\nStarting Calibration Phase. Look at the circle on the screen and press 'SPACE' for each point.")
        print("Press 'q' to quit at any time.")
        
        # --- Calibration Phase ---
        calibration_features = []
        calibration_targets_x = []
        calibration_targets_y = []
        
        calibration_window_name = "Gaze Calibration (Look at the circle)"
        cv2.namedWindow(calibration_window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(calibration_window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) # Fullscreen mode
        
        for i, (norm_x, norm_y) in enumerate(CALIBRATION_POINTS):
            target_screen_x = int(norm_x * SCREEN_WIDTH_PX)
            target_screen_y = int(norm_y * SCREEN_HEIGHT_PX)
        
            # Display initial instruction for the point
            calibration_frame_intro = np.zeros((SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX, 3), dtype=np.uint8)
            cv2.circle(calibration_frame_intro, (target_screen_x, target_screen_y), 20, (0, 255, 255), -1)
            cv2.putText(calibration_frame_intro, f"Look at {i+1}/{len(CALIBRATION_POINTS)}: Press SPACE to START collecting",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow(calibration_window_name, calibration_frame_intro)
        
            # Wait for user to press SPACE to start collecting data for this point
            while True:
                key_wait_start = cv2.waitKey(1) & 0xFF
                if key_wait_start == ord(' '):
                    break
                elif key_wait_start == ord('q'):
                    print("Calibration aborted by user.")
                    face_mesh.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            
            collected_features_for_point = []
            start_time = time.time()
        
            collected_this_point_count = 0
            while collected_this_point_count < CALIBRATION_FRAME_COUNT:
                ret, frame = cap.read() # Read a frame from the webcam
                
                if not ret:
                    print("Error: Could not read frame from webcam. Webcam might be disconnected or busy. Exiting.")
                    face_mesh.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                    #break
                
                frame = cv2.flip(frame, 1)
                # Create a blank black image for the calibration screen
                calibration_frame = np.zeros((SCREEN_HEIGHT_PX, SCREEN_WIDTH_PX, 3), dtype=np.uint8)
                cv2.circle(calibration_frame, (target_screen_x, target_screen_y), 20, (0, 255, 255), -1) # Yellow circle
                cv2.putText(calibration_frame, f"Look at {i+1}/{len(CALIBRATION_POINTS)}: Press SPACE", 
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(calibration_frame, f"Collecting... {len(calibration_features)} on {collected_this_point_count} /{CALIBRATION_FRAME_COUNT}",
                            (50, SCREEN_HEIGHT_PX - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow(calibration_window_name, calibration_frame)
                cv2.imshow('Webcam Feed (Calibration)', frame) # Show webcam feed to user (optional, can be blank)
            
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb_frame)
            
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    rotation_vector, translation_vector = get_3d_head_pose(
                        face_landmarks, image_width, image_height, camera_matrix, dist_coeffs)
            
                    if rotation_vector is not None and translation_vector is not None:
                        features = extract_gaze_features(face_landmarks, image_width, image_height, rotation_vector, translation_vector)
                        if features is not None:
                            calibration_features.append(features)
                            calibration_targets_x.append(target_screen_x)
                            calibration_targets_y.append(target_screen_y)
                            collected_this_point_count += 1 # Increment count for this point
                            #print(f"Collected {collected_this_point_count}/{CALIBRATION_FRAME_COUNT} for point {i+1}") # Feedback
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Calibration aborted by user.")
                    face_mesh.close()
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                
               
            # After collecting enough frames for this point, proceed to next
            print(f"Finished collecting data for point {i+1}/{len(CALIBRATION_POINTS)}")
            time.sleep(0.5) # Short delay before next point
        
        cv2.destroyWindow(calibration_window_name)
        cv2.destroyWindow('Webcam Feed (Calibration)')
        
        
        if not calibration_features:
            print("Not enough calibration data collected. Cannot train models. Exiting.")
            face_mesh.close()
            cap.release()
            cv2.destroyAllWindows()
            return
            
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(np.array(calibration_features)) # Fit and transform on training data
        
        # Define the parameter grid to search
        param_grid = {
            'C': [0.1, 1, 50, 100, 500, 1000], # Range for C (regularization)
            'gamma': [ 0.5, 0.1, 0.05, 0.01, 0.005, 0.001], # Range for gamma (kernel coefficient)
            'kernel': ['rbf'] # Using the Radial Basis Function kernel for non-linearity
        }
        ##############-- Train the SVM regression models ---
        print("\nTraining SVM models...")
        # Train the regression models
        # C: Regularization parameter. Higher C means less regularization (fits training data better).
        # gamma: Kernel coefficient. Higher gamma means higher influence of single training examples.
        
        #X_train = np.array(calibration_features)
        y_train_x = np.array(calibration_targets_x)
        y_train_y = np.array(calibration_targets_y)
        print("\nTraining SVM models with GridSearchCV (this may take some time)...")
        # GridSearchCV for the X model
        grid_search_x = GridSearchCV(SVR(), param_grid, refit=True, verbose=2, cv=10, n_jobs=-1) # cv=5 for 5-fold cross-validation, n_jobs=-1 uses all available cores
        grid_search_x.fit(X_train_scaled, y_train_x)
        gaze_model_x = grid_search_x.best_estimator_
        print(f"Best parameters for X model: {grid_search_x.best_params_}")
        print(f"Best score for X model: {grid_search_x.best_score_}")
        
        # GridSearchCV for the Y model
        grid_search_y = GridSearchCV(SVR(), param_grid, refit=True, verbose=2, cv=10, n_jobs=-1)
        grid_search_y.fit(X_train_scaled, y_train_y)
        gaze_model_y = grid_search_y.best_estimator_
        print(f"Best parameters for Y model: {grid_search_y.best_params_}")
        print(f"Best score for Y model: {grid_search_y.best_score_}")
        
        print("\nCalibration Complete! Evaluating Model, please wait.")
        predicted_x_train = gaze_model_x.predict(X_train_scaled)
        predicted_y_train = gaze_model_y.predict(X_train_scaled)
        
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_train_x, predicted_x_train, alpha=0.5)
        plt.plot([0, SCREEN_WIDTH_PX], [0, SCREEN_WIDTH_PX], 'r--') # Ideal y=x line
        plt.xlabel("Actual X (pixels)")
        plt.ylabel("Predicted X (pixels)")
        plt.title("X Gaze: Actual vs. Predicted (Training Data)")
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_train_y, predicted_y_train, alpha=0.5)
        plt.plot([0, SCREEN_HEIGHT_PX], [0, SCREEN_HEIGHT_PX], 'r--') # Ideal y=x line
        plt.xlabel("Actual Y (pixels)")
        plt.ylabel("Predicted Y (pixels)")
        plt.title("Y Gaze: Actual vs. Predicted (Training Data)")
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        print("\nEvaluation Complete! Starting real-time gaze tracking.") 

        
    else:
        # If model was loaded, confirm readiness for real-time tracking
        print("\nReady for real-time gaze tracking using loaded models.")
        
# --- Real-time Gaze Tracking Phase ---
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam. Exiting.")
            break        
        
        frame = cv2.flip(frame, 1) # Flip the frame horizontally for a selfie-view

        # Convert the frame from BGR to RGB (MediaPipe expects RGB input)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = None
        try:
            # Process the RGB frame with the Face Mesh model
            results = face_mesh.process(rgb_frame)
        except Exception as e:
            # Catch potential internal MediaPipe errors during processing
            print(f"Error during MediaPipe processing: {e}")
            break # Exit if MediaPipe processing itself is crashing

        gaze_text = "No face detected"
        pupil_left_coords = (None, None)
        pupil_right_coords = (None, None)
        screen_gaze_x, screen_gaze_y = None, None
        rotation_vector = None
        translation_vector = None

        if results.multi_face_landmarks:
            # Iterate through each detected face (we set max_num_faces=1, so usually just one)
            for face_landmarks in results.multi_face_landmarks:
                # Draw the full facial mesh on the original frame
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION, # Connects 468 facial landmarks
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                # Draw the specific iris connections. This is how you visually confirm pupil detection.
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_LEFT_IRIS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_RIGHT_IRIS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)
                
                # Estimate 3D Head Pose using the selected landmarks
                rotation_vector, translation_vector = get_3d_head_pose(
                    face_landmarks, image_width, image_height, camera_matrix, dist_coeffs)

                if rotation_vector is not None and translation_vector is not None:
                    # Draw the 3D head pose axes on the frame
                    frame = draw_head_pose_axes(frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    
                    # Estimate gaze direction using the 3D pupil landmarks and head pose information
                    gaze_text, pupil_left_coords, pupil_right_coords = \
                        get_gaze_direction_from_3d_pupils(
                            face_landmarks, image_width, image_height, camera_matrix, dist_coeffs, 
                            rotation_vector, translation_vector)
                    # Extract features for prediction
                    features_for_prediction = extract_gaze_features(face_landmarks, image_width, image_height, rotation_vector, translation_vector)
                    if features_for_prediction is not None:
                        # Use the calibrated models for screen gaze estimation
                        features_for_prediction_scaled = scaler.transform(np.array(features_for_prediction).reshape(1, -1))
                        screen_gaze_x, screen_gaze_y = estimate_screen_gaze_calibrated(
                            features_for_prediction_scaled[0], SCREEN_WIDTH_PX, SCREEN_HEIGHT_PX)
                    # Optional: Print head pose angles (pitch, yaw, roll) for debugging
                    # rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    # # Extract Euler angles (e.g., in radians)
                    # pitch = -np.arcsin(rotation_matrix[2,0]) # Rotation around X-axis
                    # yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0]) # Rotation around Y-axis
                    # roll = np.arctan2(rotation_matrix[2,1], rotation_matrix[2,2]) # Rotation around Z-axis
                    
                    # cv2.putText(frame, f"Pitch: {math.degrees(pitch):.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                    # cv2.putText(frame, f"Yaw: {math.degrees(yaw):.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)
                    # cv2.putText(frame, f"Roll: {math.degrees(roll):.1f} deg", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        # Display the estimated gaze direction text
        cv2.putText(frame, gaze_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        # Display the 2D pixel coordinates of the detected pupils
        cv2.putText(frame, f"Left Pupil: {pupil_left_coords}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(frame, f"Right Pupil: {pupil_right_coords}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)

    # Display Screen Gaze Coordinates
        if screen_gaze_x is not None and screen_gaze_y is not None:
            cv2.putText(frame, f"Screen Gaze: ({screen_gaze_x}, {screen_gaze_y})", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
            
            # Draw a circle on the webcam feed representing the gaze point (scaled for display)
            # This is just for visual debugging on the webcam frame.
            display_x = int(screen_gaze_x / SCREEN_WIDTH_PX * image_width)
            display_y = int(screen_gaze_y / SCREEN_HEIGHT_PX * image_height)
            cv2.circle(frame, (display_x, display_y), 5, (255, 0, 255), -1) 
        else:
            cv2.putText(frame, "Screen Gaze: N/A", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 1)
        #frame = cv2.flip(frame, 1)
        cv2.imshow('Webcam Feed (Realtime)', frame) # Use a new window name for real-time feed

        if cv2.waitKey(1) & 0xFF == ord('q'):
            face_mesh.close()
            cap.release()
            cv2.destroyAllWindows()
            store_model = input("\nDo you want to store the trained models? (y/N): ").strip().lower()
            if store_model == 'y':
                default_path = "trained_gaze_model.joblib"
                save_path = input(f"Enter path to save models (default: {default_path}): ").strip()
                if not save_path:
                    save_path = default_path
        
                # Create directory if it doesn't exist
                save_dir = os.path.dirname(save_path)
                if save_dir and not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f"Created directory: {save_dir}")
        
                try:
                    # Store both models and the scaler in a dictionary
                    model_data = {
                        'model_x': gaze_model_x,
                        'model_y': gaze_model_y,
                        'scaler': scaler
                    }
                    joblib.dump(model_data, save_path)
                    print(f"Models and scaler stored successfully at {save_path}")
                except Exception as e:
                    print(f"Error storing models: {e}")
            else:
                print("Models not stored.")
            
            break

    # Release resources before exiting
    face_mesh.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mediapipe_3d_pose_and_gaze()
