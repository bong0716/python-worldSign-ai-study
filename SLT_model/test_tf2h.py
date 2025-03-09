import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# 데이터 로드
data = np.load('landmarks_c.npy')  # 저장된 랜드마크 데이터 로드
labels = np.load('labels_c.npy')  # 저장된 라벨 데이터 로드

# 데이터와 라벨의 샘플 수 맞추기
min_samples = min(len(data), len(labels))
data = data[:min_samples]
labels = labels[:min_samples]

# 데이터 형상 확인 및 변환
print(f"Original data shape: {data.shape}")  # 데이터의 원래 형상 출력

# 데이터를 (샘플 수, 42, 3) 형상으로 변환
data = data.reshape(-1, 42, 3)
print(f"Reshaped data shape: {data.shape}")  # 변환된 데이터의 형상 출력

# 라벨을 원-핫 인코딩
lb = LabelBinarizer()
labels = lb.fit_transform(labels)  # 라벨을 원-핫 인코딩으로 변환

# 데이터 분할 (학습용/테스트용)
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# LSTM 모델 정의
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(42, 3)))  # 첫 번째 LSTM 층, 64개의 유닛, 입력 모양 설정
model.add(LSTM(64))  # 두 번째 LSTM 층, 64개의 유닛
model.add(Dense(32, activation='relu'))  # Dense 층, 32개의 유닛, ReLU 활성화 함수 사용
model.add(Dense(len(lb.classes_), activation='softmax'))  # 출력 층, 클래스 개수만큼 유닛, 소프트맥스 활성화 함수 사용

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 모델 컴파일, Adam 옵티마이저, 카테고리 손실 함수, 정확도 메트릭 사용

# 모델 학습
model.fit(X_train, y_train, epochs=30, batch_size=50, validation_data=(X_test, y_test))  # 학습 데이터로 모델 학습, 10번의 에포크, 배치 크기 32

# 모델 저장
model.save('sign_language_model_c.h5')  # 학습된 모델 저장
