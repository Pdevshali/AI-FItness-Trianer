import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import threading

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# Function to process camera feed and count squats
def count_squats():
    # Setup mediapipe instance
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # OpenCV capture from camera
    cap = cv2.VideoCapture(0)  # Use 0 for default camera

    counter = 0
    squat_started = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Make detection
        results = pose.process(image)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark

            # Get coordinates for key points
            right_hip = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
            right_knee = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
            right_ankle = np.array([landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                     landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])

            # Calculate angle
            angle = calculate_angle(right_hip, right_knee, right_ankle)

            # Counter logic
            if angle > 160 and not squat_started:
                squat_started = True
            elif angle < 100 and squat_started:
                squat_started = False
                counter += 1
                print("Squat counted:", counter)

        except:
            pass

        # Display count on the streamlit interface
        st.image(image, channels="RGB", use_column_width=True)
        st.write("Squat Count:", counter)

# Streamlit UI
def main():
    st.title("AI Fitness Trainer: Squats Analysis")

    if st.button("Start Counting"):
        # Start video capture in a separate thread
        thread = threading.Thread(target=count_squats)
        thread.start()

if __name__ == "__main__":
    main()
