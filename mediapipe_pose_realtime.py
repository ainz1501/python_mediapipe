import cv2
import mediapipe as mp

# MediaPipe Poseの初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# カメラキャプチャ
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # RGBに変換してPose処理
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ランドマークを描画
    if results.pose_landmarks:
        # ランドマークの描画
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            visibility = landmark.visibility
            color = (0, 0, 255) if visibility >= 0.5 else (255, 0, 0)  # 赤 (BGR: 0,0,255) or 青 (255,0,0)
            cv2.circle(frame, (x, y), 5, color, -1)
        
        # 接続線の描画
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start_landmark = results.pose_landmarks.landmark[start_idx]
            end_landmark = results.pose_landmarks.landmark[end_idx]
            start_vis = start_landmark.visibility
            end_vis = end_landmark.visibility
            avg_vis = (start_vis + end_vis) / 2
            color = (0, 0, 255) if avg_vis >= 0.5 else (255, 0, 0)
            start_point = (int(start_landmark.x * frame.shape[1]), int(start_landmark.y * frame.shape[0]))
            end_point = (int(end_landmark.x * frame.shape[1]), int(end_landmark.y * frame.shape[0]))
            cv2.line(frame, start_point, end_point, color, 2)

    # 表示
    cv2.imshow('MediaPipe Pose', frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# クリーンアップ
cap.release()
cv2.destroyAllWindows()