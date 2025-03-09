# openai_chat.py
import openai

class OpenAIChat:
    def __init__(self, api_key, model="gpt-4o-mini", system_message="한글 단어를 한국어로 번역 후 자연스러운 한국어 문장으로 만들어줘"):
        """
        OpenAIChat 클래스의 생성자.
        :param api_key: OpenAI API 키
        :param model: 사용할 OpenAI 모델
        :param system_message: 시스템 메시지
        """
        openai.api_key = api_key
        self.model = model
        self.system_message = system_message

    def generate_response(self, user_message):
        """
        사용자의 메시지를 기반으로 OpenAI API를 호출하여 응답을 생성합니다.
        :param user_message: 사용자의 메시지
        :return: 응답 내용
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message['content']
