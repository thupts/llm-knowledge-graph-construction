import streamlit as st
from utils import write_message
from agent import generate_response

st.set_page_config("Ebert", page_icon="🎙️")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
    ]

# Submit handler
def handle_submit(message: str):
    """
    Submit handler: 
    - lưu tin nhắn user,
    - gọi agent (LLM + Neo4j tools),
    - lưu và hiển thị tin nhắn assistant.
    """
    # Lưu tin nhắn user vào session state
    st.session_state.messages.append({"role": "user", "content": message})

    # Gọi agent để sinh câu trả lời
    with st.spinner("Thinking..."):
        try:
            response = generate_response(message)
        except Exception as e:
            response = f"⚠️ Error: {str(e)}"

    # Lưu tin nhắn assistant
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Hiển thị tin nhắn assistant
    write_message("assistant", response)

# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
