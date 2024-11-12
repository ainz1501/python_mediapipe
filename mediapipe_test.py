import cv2
import mediapipe as mp
import math
from PIL import Image

# Mediapipeの初期化
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# ウェブカメラのキャプチャを開始
cap = cv2.VideoCapture(0)
# 画像入力

#　予測のための環境情報を入力
hand_length = 18.5
# distance = 100

def predict_distance1(distance, hand_length):
    predict_landmarks_z = -hand_length/(2*distance)
    cv2.putText(image,f"Landmark[12] Distance:{predict_landmarks_z:4f} ", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return predict_landmarks_z

def predict_distance2(distance, hand_length):
    a = hand_length*(math.sqrt(3)/2)
    b = distance - (hand_length/2)
    midtip_distance = math.sqrt(pow(a, 2)+pow(b, 2))
    premidtip_z = 1-(midtip_distance/distance)
    print(f"a:{a} b:{b}")
    cv2.putText(image,f"Landmark[12] Distance:{premidtip_z:4f} ", (20, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    return premidtip_z

def print_error(img, prelandmarks_z, midtip_z):
    Landmarks_error = prelandmarks_z - midtip_z
    cv2.putText(img,f"Landmark[12].z:{midtip_z:4f} ", (20, 60), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(image,f"Error:{Landmarks_error:4f} ", (20, 90), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2, cv2.LINE_AA)


while(1):
    # 手の全長[cm]
    # hand_length = int(input("手の全長を入力[cm]："))
    # 測定距離[cm]
    distance = float(input("測定する手首の距離を入力[cm]："))
    if isinstance(hand_length, (int, float)) and isinstance(distance, (int, float)):
        break
    print("数値ではありません\n")

# Hand Landmark Detectionを設定
with mp_hands.Hands(
    model_complexity=0,  # 0に設定するとより高速化できる
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=1) as hands:

    # webカメラ起動、画像キャプチャ開始
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラからフレームを取得できませんでした")
            continue

        # 画像を反転して、結果をより自然に表示
        image = cv2.flip(image, 1)

        # BGR画像をRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画面中央に赤点を描写
        height, width, _ = image.shape
        img_center = (width//2, height//2)
        cv2.circle(image, img_center, 5, (0, 0, 255), thickness=-1)

        # 画像をMediapipeに渡して処理
        results = hands.process(image_rgb)
        
        # 中指の指先のzの値予想を表示1
        pre_z1 = predict_distance1(distance, hand_length)
        # pre_z2 = predict_distance2(distance, hand_length)
        
        # 検出された手のランドマークを描画
        if results.multi_hand_landmarks:
            # 中指先のzを取得
            midtip_z = results.multi_hand_landmarks[0].landmark[12].z
            for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # predict_distance2()

                # 誤差を表示
                print_error(image, pre_z1, midtip_z)
                # print_error(image, pre_z2, midtip_z)


        # 画像を表示
        cv2.imshow('Hand Landmarks', image)

        # 'q'キーが押されたら終了
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# リソースを解放
cap.release()
cv2.destroyAllWindows()