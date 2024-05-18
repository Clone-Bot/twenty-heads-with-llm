import warnings

warnings.filterwarnings("ignore")

import gradio as gr

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage

# AI가 질문
llm = ChatUpstage()

num_count = 5
# More general chat
chat_with_history_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"""
You have to narrow the scope by asking questions in Korean to try to guess the specific word the user had in mind.
The rules are as follows.
1. You should never ask a question that is similar to a question you once asked.
2. You must find the word you were thinking of by asking a total of {num_count} questions, so you must be careful about each question.
3. In front of every question, write the number of times you asked the question.
4. If you ask a question more than {num_count} times, you must say the correct answer you think.
5. And please give all answers only in Korean.
6. Please answer only one question at a time.
7. Answer everything except the question you ask the user and your answer with "...".
8. Put a question mark at the end of the question.
9. No words can come after the question mark.
          """),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)

chain = chat_with_history_prompt | llm | StrOutputParser()

def chat(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    return chain.invoke({"message": message, "history": history_langchain_format})

def init_quiz():
    print(1)
    pass

with gr.Blocks() as demo:

    demo.load(init_quiz)

    chatbot = gr.ChatInterface(
        chat,
        examples=[
            "너가 생각한건 동물이야?",
            "그건 먹을 수 있니?",
        ],
        title="스무고개",
        description="Solar 를 이용한 스무고개 챗봇",
    )
    chatbot.chatbot.height = 500

if __name__ == "__main__":
    demo.launch()
    