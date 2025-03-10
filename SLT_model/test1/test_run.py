import cv2
import mediapipe as mp
import numpy as np
import os

# Mediapipe 손 추적기 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# 데이터와 라벨을 저장할 리스트
data = []
labels = []

# 데이터 디렉토리 설정
data_dir = 'slt_v'  # 수어 단어 영상이 저장된 디렉토리 경로

# 클래스 라벨 설정 (예: 'hello', 'thank_you', ...)
classes = ['옆쪽', '오늘', '화장실', '화재']  # 수어 단어 클래스 리스트

for label in classes:
    class_dir = os.path.join(data_dir, label)  # 각 클래스(수어 단어)의 디렉토리
    for video_file in os.listdir(class_dir):
        if video_file.endswith('.avi'):  # .mp4 확장자를 가진 비디오 파일만 처리
            cap = cv2.VideoCapture(os.path.join(class_dir, video_file))  # 비디오 파일 열기

            while cap.isOpened():
                ret, frame = cap.read()  # 비디오의 한 프레임 읽기
                if not ret:
                    break  # 더 이상 읽을 프레임이 없으면 종료

                # 프레임을 Mediapipe로 처리
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
                result = hands.process(frame_rgb)  # 손 랜드마크 인식 수행

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        # 손 랜드마크 추출
                        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                        data.append(landmarks.flatten())  # 랜드마크를 1차원 배열로 변환하여 저장
                        labels.append(classes.index(label))  # 해당 클래스의 인덱스를 라벨로 저장

            cap.release()  # 비디오 파일 닫기

data = np.array(data)  # 데이터 배열로 변환
labels = np.array(labels)  # 라벨 배열로 변환

# 데이터 저장 (필요 시)
np.save('landmarks.npy', data)  # 랜드마크 데이터를 numpy 파일로 저장
np.save('labels.npy', labels)  # 라벨 데이터를 numpy 파일로 저장
