import cv2
import mediapipe as mp

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 画像を読み込む
image_path = '/Users/tokudataichi/Documents/python_mediapipe/IMG_4997.JPG'
image = cv2.imread(image_path)
image2_path = '/Users/tokudataichi/Documents/python_mediapipe/IMG_4998.JPG'
image2 = cv2.imread(image2_path)

# BGRをRGBに変換（MediaPipeがRGBを期待するため）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# 平行ステレオビジョン
def stereo_vision_easy():
    holistic_3d_landmarks =[46][3]
    baseline = 0.5

# Holisticモジュールを使って画像を処理
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(image_rgb)
    results2 = holistic.process(image2_rgb)

    # 結果の可視化
    if results.pose_landmarks: # 画像１の結果
        print("pose through")
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0))) 
    if results.left_hand_landmarks:
        print("left hand through")
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    if results.right_hand_landmarks:
        print("right hand through")
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    
    if results2.pose_landmarks: # 画像２の結果
        mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))  
    if results2.left_hand_landmarks: 
        mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    if results2.right_hand_landmarks:
        mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))        
        

# 処理結果を表示
cv2.imshow('Holistic Result', image)
cv2.imshow('Holistic Result 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()