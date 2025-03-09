import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Mediapipe 손 추적기 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
model = tf.keras.models.load_model('sign_language_model.h5')

# 클래스 라벨 설정 (모델 학습 시 사용했던 순서대로)
classes = ['옆쪽', '오늘', '화장실', '화재']  # 수어 단어 클래스 리스트

# 웹캠 캡처 시작
cap = cv2.VideoCapture(1)  # 웹캠 ID를 0으로 변경 (필요에 따라 조정)

while cap.isOpened():
    ret, frame = cap.read()  # 웹캠의 한 프레임 읽기
    if not ret:
        break

    # 프레임을 Mediapipe로 처리
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
    result = hands.process(frame_rgb)  # 손 랜드마크 인식 수행

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 손 랜드마크 추출
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            landmarks = landmarks.reshape(1, 21, 3)  # 모델 입력 형식에 맞게 변환

            # 모델 예측 수행
            prediction = model.predict(landmarks)
            predicted_class = np.argmax(prediction)
            predicted_label = classes[predicted_class]

            # 결과를 프레임에 표시
            cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            print(predicted_label)

            # 랜드마크를 프레임에 그리기
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 프레임 출력
    cv2.imshow('Sign Language Recognition', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
