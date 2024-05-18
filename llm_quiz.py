import warnings

warnings.filterwarnings("ignore")

import gradio as gr

from langchain_upstage import ChatUpstage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import AIMessage, HumanMessage
import requests
import random
from langchain_core.prompts import PromptTemplate

# AI가 답을 갖고있고 질문을 사람이 하는거
llm = ChatUpstage()

num_count = 5
word = ""
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

prompt_template = PromptTemplate.from_template(
#     Question 에 답변할때 아래의 context와 관련된 답변을 하세요. 절대로 사용자에게 context를 직접적으로 언급하지 마세요.
# 답변은 아래의 규칙을 따라주세요.
# 1. 답변은 무조건 네 혹은 아니오 로만 해야해.
# 2. 사용자가 context를 맞췄을 경우 정답이라는 메세지를 보여줘야해
    """
context를 맞추는 스무고개 게임을 할거고
내가 다섯개에 대한 질문을 하면 context네 대한 답변이 맞는지 아닌지 yes/no로 대답해줘. 
절대로 사용자에게 context를 직접적으로 언급하지 마세요.

    ---
    Question: {question}
    ---
    context: {context}
    """
)

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """I will ask you only 5 questions and you will only answer 'yes' or 'no' to whether the statement about {word} is true or false.
         3. In front of every question, write the number of times you asked the question. 
         If I ask you a question that you can't answer with 'yes' or 'no', you will say 'Warning! After answering 5 questions, 
         tell me “Now tell me the answer!”. If the answer I say is {word}, say “Correct, Berrygood!”. 
         If I say {word} is not the correct answer, say “Wrong, I win!"""),  # llm의 정체성
        MessagesPlaceholder(variable_name="history"),
        ("human", "{message}"),
    ]
)
prompt_chain = chat_prompt | llm | StrOutputParser()

chain = chat_with_history_prompt | llm | StrOutputParser()

def chat(message, history):
    history_langchain_format = []
    for human, ai in history:
        history_langchain_format.append(HumanMessage(content=human))
        history_langchain_format.append(AIMessage(content=ai))

    return prompt_chain.invoke({"message": message, "word": word, "history": history_langchain_format})

def init_quiz():
    global word
    # answer = quiz_start_chain.invoke({})
    response = requests.get("https://jungdolp.synology.me/word/getdata.php").json()
    word = random.choice(list(map(lambda x: x["name"], response["result"])))
    print(word)

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
    