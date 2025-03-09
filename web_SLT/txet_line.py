from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from time import sleep
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import defaultdict
from openai_chat import OpenAIChat

app = Flask(__name__)
socketio = SocketIO(app)
capture = cv2.VideoCapture(1)  # 웹캠으로부터 비디오 캡처 객체 생성
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 캡처된 비디오의 폭 설정
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 캡처된 비디오의 높이 설정

# Mediapipe 손 추적기 설정
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 학습된 모델 로드
model = tf.keras.models.load_model('sign_language_model_k.h5')

# 클래스 라벨 설정 (모델 학습 시 사용했던 순서대로)
classes_k = ['고생하다', '만나다', '반갑다', '여러분', '오늘']
classes_e = ['hello', 'sorry','thankyou']
classes_c = ['不客气', '你好', '谢谢']
classes =  classes_k # 수어 단어 클래스 리스트

# 인식된 단어와 해당 단어의 검출 횟수 저장
detected_words = defaultdict(int)
recognized_words = set()

# OpenAIChat 인스턴스 생성
openai_chat = OpenAIChat(api_key="api-key")  # 여기에 실제 OpenAI API 키를 입력하세요.

def GenerateFrames():
    while True:
        # sleep(0.1)  # 프레임 생성 간격을 잠시 지연시킵니다.
        ref, frame = capture.read()  # 비디오 프레임을 읽어옵니다.
        if not ref:  # 비디오 프레임을 제대로 읽어오지 못했다면 반복문을 종료합니다.
            break
        else:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 프레임을 RGB로 변환
            result = hands.process(frame_rgb)  # 손 랜드마크 인식 수행

            predicted_label = "변역할 언어를 선택하세요"  # 기본값

            if result.multi_hand_landmarks:
                # 양손의 랜드마크를 저장할 배열
                left_hand_landmarks = np.zeros((21, 3))
                right_hand_landmarks = np.zeros((21, 3))

                for hand_landmarks, hand_label in zip(result.multi_hand_landmarks, result.multi_handedness):
                    handedness = hand_label.classification[0].label

                    # 왼손과 오른손을 구분하여 저장
                    if handedness == 'Left':
                        left_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
                    elif handedness == 'Right':
                        right_hand_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

                # 양손 랜드마크를 하나의 배열로 결합
                landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
                landmarks = landmarks.reshape(1, 42, 3)  # 모델 입력 형식에 맞게 변환

                # 모델 예측 수행
                prediction = model.predict(landmarks)
                predicted_class = np.argmax(prediction)
                predicted_label = classes[predicted_class]

                # 검출된 단어 횟수 증가
                detected_words[predicted_label] += 1

                # 검출 횟수가 10 이상이면 리스트에 포함
                if detected_words[predicted_label] >= 10:
                    recognized_words.add(predicted_label)

                # 결과를 프레임에 표시
                # cv2.putText(frame, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(predicted_label)

                # 랜드마크를 프레임에 그리기
                # for hand_landmarks in result.multi_hand_landmarks:
                #     mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 이미지 데이터를 JPEG 포맷으로 인코딩
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # 웹소켓을 통해 예측된 레이블을 전송
            handle_get_detected_words()

            # multipart/x-mixed-replace 포맷으로 비디오 프레임을 클라이언트에게 반환합니다.
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            
def handle_get_detected_words():
    # detected_words 리스트를 클라이언트로 전송
    words = list(recognized_words)
    words_list = '\n'.join(words)
    socketio.emit('detected_words_response', {'detected_words': words_list})

@app.route('/')
def Index():
    return render_template('index.html')  # index.html 파일을 렌더링하여 반환합니다.


@app.route('/stream')
def Stream():
    # GenerateFrames 함수를 통해 비디오 프레임을 클라이언트에게 실시간으로 반환합니다.
    return Response(GenerateFrames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('nationalities')
def handle_get_nationality_words(data):
    global model, classes, classes_k, classes_e, classes_c, detected_words, recognized_words
    # 번역할 언어 설정
    nationality1 = data.get('nationality1')
    nationality2 = data.get('nationality2')
    if nationality1 == '한국어':
        model = tf.keras.models.load_model('sign_language_model_k.h5')
        classes = classes_k
    elif nationality1 == '영어':
        model = tf.keras.models.load_model('sign_language_model_e.h5')
        classes = classes_e
    elif nationality1 == '중국어':
        model = tf.keras.models.load_model('sign_language_model_c.h5')
        classes = classes_c
    openai_chat.set_language(nationality1,nationality2)

    # detected_words와 recognized_words 초기화
    detected_words = defaultdict(int)
    recognized_words = set()
    


@socketio.on('generate_sentence')
def handle_generate_sentence():
    global detected_words, recognized_words

    # 인식된 단어들을 하나의 문장으로 생성
    if recognized_words:
        user_message = ' '.join(recognized_words)
        response = openai_chat.generate_response(user_message)
        sentence = response
    else:
        sentence = "No words recognized."

    # detected_words와 recognized_words 초기화
    detected_words = defaultdict(int)
    recognized_words = set()

    # 생성된 문장을 클라이언트로 전송
    socketio.emit('generated_sentence_response', {'sentence': sentence})

@socketio.on('reset_list')
def reset_list():
    global detected_words, recognized_words

    # detected_words와 recognized_words 초기화
    detected_words = defaultdict(int)
    recognized_words = set()

if __name__ == "__main__":
    # Flask 앱을 실행합니다.
    socketio.run(app, host="host", port=port)
