# Import the Streamlit library for creating web applications.
import streamlit as st
# Import the OS library to interact with the operating system, such as reading environment variables.
import os
# Import the Groq module, presumably for API interactions (though it's not standard and may be custom or from a specific library not detailed here).
from groq import Groq
# Import the random library to generate random numbers, sequences, or select items at random.
import random

# Import ConversationChain for creating a chain of conversation logic.
from langchain.chains import ConversationChain
# Import ConversationBufferWindowMemory for managing conversational memory in a sliding window fashion.
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
# Import ChatGroq from langchain_groq for integrating Groq API into the conversation logic.
from langchain_groq import ChatGroq
# Import PromptTemplate, which is likely used for formatting or templating prompts in conversations.
from langchain.prompts import PromptTemplate
# Import load_dotenv function from dotenv to load environment variables from a .env file.
from dotenv import load_dotenv
# Re-import os which is redundant and could be removed since it's already imported above.
import os 

# Call load_dotenv function to load the environment variables from a .env file into the OS environment.
load_dotenv()

# Retrieve the 'GROQ_API_KEY' environment variable and store it in a variable for later use.
groq_api_key = os.environ['GROQ_API_KEY']

# Define the main function that contains the logic of the web app.
def main():
    # Set the title of the web application.
    st.title("Groq Chat App")

    # Create a sidebar section titled 'Select an LLM' for model selection.
    st.sidebar.title('Select an LLM')
    # Create a dropdown menu in the sidebar for selecting a model, with two options.
    model = st.sidebar.selectbox(
        'Choose a model',
        ['mixtral-8x7b-32768', 'llama2-70b-4096']
    )
    # Create a slider in the sidebar for selecting the conversational memory length, ranging from 1 to 10 with a default value of 5.
    conversational_memory_length = st.sidebar.slider('Conversational memory length:', 1, 10, value=5)

    # Initialize conversational memory with the selected length using ConversationBufferWindowMemory.
    memory=ConversationBufferWindowMemory(k=conversational_memory_length)

    # Create a text area in the main body of the app for the user to input their question.
    user_question = st.text_area("Ask a question:")

    # Initialize or maintain a chat history in the session state to persist across reloads.
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history=[]
    else:
        # If chat history exists, load previous conversations into the memory for context awareness.
        for message in st.session_state.chat_history:
            memory.save_context({'input':message['human']},{'output':message['AI']})

    # Initialize the ChatGroq object with the API key and selected model for handling conversation logic.
    groq_chat = ChatGroq(
            groq_api_key=groq_api_key, 
            model_name=model
    )

    # Initialize the conversation chain with the ChatGroq object and the conversational memory.
    conversation = ConversationChain(
            llm=groq_chat,
            memory=memory
    )

    # If the user submits a question, process the conversation and display the response.
    if user_question:
        response = conversation(user_question)
        # Store the user question and AI response in a dictionary format.
        message = {'human':user_question,'AI':response['response']}
        # Append the conversation to the session state's chat history.
        st.session_state.chat_history.append(message)
        # Display the AI's response in the web application.
        st.write("Chatbot:", response['response'])

# Check if the script is executed as the main program and not imported as a module, then run the main function.
if __name__ == "__main__":
    main()
