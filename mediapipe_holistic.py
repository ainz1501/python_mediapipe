import cv2
import mediapipe as mp

# MediaPipe Holisticモジュールを初期化
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 画像を読み込む サイズは4284 × 5712
image_path = '/Users/tokudataichi/Documents/python_mediapipe/left0_image.JPG'
image = cv2.imread(image_path)
image2_path = '/Users/tokudataichi/Documents/python_mediapipe/right50_image.JPG'
image2 = cv2.imread(image2_path)

# BGRをRGBに変換（MediaPipeがRGBを期待するため）
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

# カメラ座標版キーポイント構造体
class cam_landmark:
    x:float
    y:float
    z:float = 0.0

    def __init__(self, input_x, input_y) -> None:
        self.x = input_x
        self.y = input_y

# 平行ステレオビジョン
# def stereo_vision_easy(result, result2):
#     # holistic_3d_landmarks =[46][3]
#     pose_3d_landmarks = [33]
#     right_hand_3d = [21]
#     left_hand_3d = [21]
#     baseline = 0.5

#     result_camera = transform2camera(result)
#     result2_camera = transform2camera(result2)

#     if result.pose_landmarks and result2.pose_landmarks:

# カメラ座標への変換（同一デバイス使用前提）
def transform2camera(result, width, height):
    cam_landmarks = [] # カメラ座標を記録する配列を生成
    for index, landmark in enumerate(result.landmark):
        # キーポイントの値をカメラ座標系（画像中央が原点）に変換
        cam_x = landmark.x * width - width/2.0
        cam_y = landmark.y * height - height/2.0
        # cam_landmarkのインスタンスの生成と同時に、x座標、y座標の値を入力
        cam_landmarks.append(cam_landmark(cam_x, cam_y)) 

        # print(f"cam_landmarks[{index}].x: {cam_landmarks[index].x}")

    return cam_landmarks
        
def transform_result(results, image):
    height, width, _ = image.shape

    if results.pose_landmarks:
        pose_cam_landmarks = transform2camera(results.pose_landmarks, width, height)
        for index, cam_landmarks in enumerate(pose_cam_landmarks):
            print(f"pose_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        pose_cam_landmarks = None
    if results.left_hand_landmarks:
        left_hand_cam_landmarks = transform2camera(results.left_hand_landmarks, width, height)
        for index, cam_landmarks in enumerate(left_hand_cam_landmarks):
            print(f"left_hand_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        left_hand_cam_landmarks = None
    if results.right_hand_landmarks:
        right_hand_cam_landmarks = transform2camera(results.right_hand_landmarks, width, height)
        for index, cam_landmarks in enumerate(right_hand_cam_landmarks):
            print(f"right_hand_cam_landmarks[{index}].x : {cam_landmarks.x}")
    else:
        right_hand_cam_landmarks = None
    
    return pose_cam_landmarks, left_hand_cam_landmarks, right_hand_cam_landmarks

# Holisticモジュールを使って画像を処理
with mp_holistic.Holistic(static_image_mode=True) as holistic:
    results = holistic.process(image_rgb)
    results2 = holistic.process(image2_rgb)

    # 結果の可視化
    if results.pose_landmarks: # 画像１のレンダリング
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
    
    if results2.pose_landmarks: # 画像２のレンダリング
        print("pose 2 through")
        mp_drawing.draw_landmarks(image2, results2.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))  
    if results2.left_hand_landmarks: 
        print("left hand 2 through")
        mp_drawing.draw_landmarks(image2, results2.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))
    if results2.right_hand_landmarks:
        print("right hand 2 through")
        mp_drawing.draw_landmarks(image2, results2.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                  mp_drawing.DrawingSpec(color=(0,0,255)), mp_drawing.DrawingSpec(color=(0,0,0)))  
    
    # 出力したキーポイントをカメラ座標系に変換(x,yの値のみ)
    pose_cam, left_hand_cam, right_hand_cam = transform_result(results, image) 
    pose_cam2, left_hand_cam2, right_hand_cam2 = transform_result(results2, image2)     
        

# 処理結果を表示
cv2.imshow('Holistic Result', image)
cv2.imshow('Holistic Result 2', image2)
cv2.waitKey(0)
cv2.destroyAllWindows()