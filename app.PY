'''!pip install langchain langchain-community
!pip install langchain-openai
!pip install gradio'''
import os
import sys
import json
from langchain_openai import ChatOpenAI
import random
import gradio as gr

MODEL = "gpt-4o-mini"
API_URL = os.getenv("API_URL")
OPENAI_API_KEYS = os.getenv("OPENAI_API_KEYS", "").split(",") if os.getenv("OPENAI_API_KEYS") else []
NUM_THREADS = int(os.getenv("NUM_THREADS", 1))

DEFAULT_API_KEY = "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"

def exception_handler(exception_type, exception, traceback):
    print("%s: %s" % (exception_type.__name__, exception))
sys.excepthook = exception_handler
sys.tracebacklimit = 0

class ChatBot:
    def __init__(self):
        self.api_key = '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'

        self.chat = ChatOpenAI(
            api_key=self.api_key,
            base_url="https://models.inference.ai.azure.com",
            model='gpt-4o',
            streaming=True,
            temperature=1.0,
            top_p=1.0
        )
        self.history = []
        self.chat_counter = 0

    def predict(self, user_input, temperature=1.0, top_p=1.0):
        self.chat.temperature = temperature
        self.chat.top_p = top_p

        try:
            messages = []
            for i, content in enumerate(self.history):
                message = {
                    "role": "user" if i % 2 == 0 else "assistant",
                    "content": content
                }
                messages.append(message)

            messages.append({
                "role": "user",
                "content": user_input
            })

            response = self.chat.invoke(messages)
            assistant_response = response.content

            self.history.append(user_input)
            self.history.append(assistant_response)
            self.chat_counter += 1

            return assistant_response

        except Exception as e:
            print(f'Error found: {e}')
            return str(e)

    def reset(self):
        self.history = []
        self.chat_counter = 0
        return ""

chatbot_instance = ChatBot()

def chatbot_response(message, history):
    response = chatbot_instance.predict(message)
    return history + [[message, response]]

def clear_history():
    chatbot_instance.reset()
    return None

# Create Gradio interface
def main():
    with gr.Blocks(css="footer {visibility: hidden}") as demo:
        gr.Markdown("#                SONU KUMAR Chatbot")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Type your message here...", placeholder="Type your message and press Enter...")
        clear = gr.Button("Clear")

        # Set up the message submission interaction
        msg.submit(
            chatbot_response,
            inputs=[msg, chatbot],
            outputs=[chatbot],
        ).then(
            lambda: "",
            None,
            msg,
            queue=False
        )

        # Set up the clear button interaction
        clear.click(
            clear_history,
            outputs=[chatbot],
            queue=False
        )

    # Launch the interface
    demo.launch(share=True)

if __name__ == "__main__":
    main()