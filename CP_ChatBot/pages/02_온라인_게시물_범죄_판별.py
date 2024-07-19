import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.callbacks.base import BaseCallbackHandler

# api_key.py
def get_api_key():
    with open("api_key.txt", "r") as file:
        return file.read().strip()
api_key = get_api_key()

# 페이지 제목
st.title("온라인 게시물 범죄 판별기")

# 사용자로부터 게시물 입력 받기
st.header("게시물 입력")
user_input = st.text_area("여기에 게시물을 입력하세요", height=200)
class ChatCallbackHandler(BaseCallbackHandler):
    message = ""
    
    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()
        
    
    def on_llm_new_token(self, token: str, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(
    api_key=api_key,
    temperature=0.1,
    model_name="gpt-4-turbo",
    streaming=True,
    callbacks=[ChatCallbackHandler(),],
    
)


# # 게시물 분류 버튼
# if st.button("게시물 분류"):
#     if user_input:
#         # 예시 분류 및 위험도
#         classification = "성범죄"  # 성범죄, 마약, 해킹, 일반 중 하나로 분류
#         risk_percentage = 85  # 위험도 퍼센트
#         penalty_info = "성범죄로 분류된 게시물은 최대 5년의 징역형에 처해질 수 있습니다."

#         # 분류 결과 출력
#         st.subheader("분류 결과")
#         st.write(f"분류: {classification}")

#         # 위험도 출력 (일반이 아닌 경우에만)
#         if classification != "일반":
#             st.write(f"위험도: {risk_percentage}%")

#             # 처벌 정보 출력
#             st.subheader("관련 처벌 정보")
#             st.write(penalty_info)
#     else:
#         st.warning("게시물을 입력해주세요.")

# 예시 데이터
sample_data = {
    "성범죄": {"위험도": 85, "처벌": "성범죄로 분류된 게시물은 최대 5년의 징역형에 처해질 수 있습니다."},
    "마약": {"위험도": 70, "처벌": "마약 관련 게시물은 최대 3년의 징역형에 처해질 수 있습니다."},
    "해킹": {"위험도": 90, "처벌": "해킹 관련 게시물은 최대 10년의 징역형에 처해질 수 있습니다."},
    "일반": {"위험도": 0, "처벌": "해당 게시물은 범죄와 관련이 없습니다."},
}

# 예시 게시물 분류 및 처벌 정보 표시
st.sidebar.header("예시 게시물")
example_option = st.sidebar.selectbox("예시 게시물 선택", list(sample_data.keys()))

if example_option:
    example_risk = sample_data[example_option]["위험도"]
    example_penalty = sample_data[example_option]["처벌"]

    st.sidebar.subheader("분류 결과")
    st.sidebar.write(f"분류: {example_option}")
    if example_option != "일반":
        st.sidebar.write(f"위험도: {example_risk}%")
        st.sidebar.subheader("관련 처벌 정보")
        st.sidebar.write(example_penalty)

if st.button("게시물 분류"):
    if user_input:
        # 예시 분류 및 위험도
        classification = "성범죄"  # 성범죄, 마약, 해킹, 일반 중 하나로 분류
        risk_percentage = 81  # 위험도 퍼센트
        penalty_info = "성범죄로 분류된 게시물은 최대 5년의 징역형에 처해질 수 있습니다."

        # 분류 결과 출력
        st.subheader("분류 결과")
        st.write(f"분류: {classification}")

        # 위험도 출력 (일반이 아닌 경우에만)
        if classification != "일반":
            st.write(f"위험도: {risk_percentage}%")

           # 처벌 정보 출력
            st.subheader("게시물 처벌 정보")
            intent_prompt = ChatPromptTemplate.from_messages([
                ("system", """
                    당신은 {classification} 관련 범죄 게시물의 의도를 파악한 후 의도에 따라 행동할 경우 형벌을 예상하는 ai 입니다.
                    형벌의 근거 법룰을 언급해하여 대답해주세요.

                """),
                ("human", "{content}")
            ])
            
            intent_chain = {
                "classification": RunnablePassthrough(),
                "content":RunnablePassthrough()
            } | intent_prompt | llm
            
            
            intent_result = intent_chain.invoke({"classification": classification, "content":user_input })
            
            st.markdown(intent_result.content)
            
            
    else:   
        st.warning("게시물을 입력해주세요.")