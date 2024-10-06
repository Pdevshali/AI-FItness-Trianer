import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Function to calculate angle between three points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle
    

def main():
    # Initialize variables
    squat_started = False
    counter = 0
    improper_form_warning = False

    # Setup mediapipe instance
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Streamlit UI
    st.title("AI Squat Counter")

    # Video capture setup
    cap = cv2.VideoCapture("sqt.mp4")

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for key points
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                         landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
            right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

            # Calculate angle
            angle = calculate_angle(np.array(right_shoulder), np.array(right_hip), np.array(right_knee))

            # Counter logic
            if angle > 160 and not squat_started:
                squat_started = True
            elif angle < 100 and squat_started:
                squat_started = False
                counter += 1

            # Check for improper squat form
            MIN_ACCEPTABLE_ANGLE = 30
            MAX_ACCEPTABLE_ANGLE = 180
            if angle < MIN_ACCEPTABLE_ANGLE or angle > MAX_ACCEPTABLE_ANGLE:
                improper_form_warning = True
            else:
                improper_form_warning = False

        except:
            pass

        # Display warning if improper form detected
        if improper_form_warning:
            st.warning("Please ensure proper squat form!")

        # Display count
        st.write("Count: ", counter)

        # Display image
        st.image(image, channels="BGR")

    cap.release()

if __name__ == "__main__":
    main()
