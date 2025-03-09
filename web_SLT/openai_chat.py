import openai

class OpenAIChat:
    def __init__(self, api_key, model="gpt-4"):
        """
        OpenAIChat 클래스의 생성자.
        :param api_key: OpenAI API 키
        :param model: 사용할 OpenAI 모델
        :param system_message: 시스템 메시지
        """
        openai.api_key = api_key
        self.model = model
        self.system_message = '''너는 수어 통역 전문가야. 이제 부터 입력될 단어들을 수어를 검출한 단어들이야. 입련된 단어들을 자면스러운 문장으로 만들어줘. 입력받은 단어가 영어일 겅우 한국어로 번역 후 문장을 만들고 한국어일 경우 영어로 번역해서 문장을 만들어줘.'''

    def set_language(self, in_lang, out_lang):
        self.system_message = f'''너는 수어 통역 전문가야. 이제 부터 입력될 단어들을 수어를 검출한 단어들이야. 입련된 단어들을 자면스러운 문장으로 만들어줘. 입력받은 {in_lang}단어를 {out_lang}로 번역해서 문장을 만들어줘.'''


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
