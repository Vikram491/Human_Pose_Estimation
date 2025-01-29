import mediapipe as mp
import cv2

mp_pose = mp.solutions.pose  
mp_drawing = mp.solutions.drawing_utils

# Set video source: 0 for webcam, or provide a video file path (e.g., "dance1.mp4")
video_source = 0
#cap = cv2.VideoCapture(video_source)
cap = cv2.VideoCapture("dance1.mp4")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' codec for MP4 files
out = cv2.VideoWriter('Pose_Estimation_Output.mp4', fourcc, 30, (960, 540))

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=0,  # Use the simplest model for faster processing
    enable_segmentation=False,  # Disable segmentation (not used here)
    min_detection_confidence=0.5,  # Minimum confidence to detect pose landmarks
    min_tracking_confidence=0.5  # Minimum confidence to track pose landmarks
) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended")
            break
        
        frame = cv2.flip(frame, 1)

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        res = pose.process(frame_rgb)

        # If pose landmarks are detected, draw them on the frame
        if res.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, res.pose_landmarks,  # The detected pose landmarks
                mp_pose.POSE_CONNECTIONS,  # Connections between the landmarks (skeleton lines)
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
            )

        display_frame = cv2.resize(frame, (960, 540))

        out.write(display_frame)

        cv2.imshow('Pose Estimation', display_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
out.release()
cv2.destroyAllWindows()
