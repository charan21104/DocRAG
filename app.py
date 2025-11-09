import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai import ChatGoogleGenerativeAI
import asyncio

# This is the correct, cross-platform fix for the "no current event loop" error
try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


# This function creates the vector store using free HuggingFace embeddings.
def get_vectorstore(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore

# This function creates the conversation chain using a model confirmed to be available to your API key.
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-pro-latest", convert_system_message_to_human=True)

    # Initialize memory
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# --- THIS FUNCTION HAS BEEN CORRECTED ---
def handle_userinput(user_question):
    if st.session_state.conversation:
        # The chain expects a dictionary with 'question' and 'chat_history' keys
        response = st.session_state.conversation({
            'question': user_question,
            'chat_history': st.session_state.chat_history or [] # Ensure history is a list
        })
        st.session_state.chat_history = response['chat_history']

        # Loop through the updated chat history to display it
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.chat_message("user").write(message.content)
            else:
                st.chat_message("assistant").write(message.content)
    else:
        st.warning("Please upload and process documents first.")

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    st.header("Chat with your PDFs :books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [] # Initialize as an empty list

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs and click 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    vectorstore = get_vectorstore(pdf_docs)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Done!")
            else:
                st.warning("Please upload at least one PDF.")
    
    # Display existing chat history
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    user_question = st.chat_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

if __name__ == '__main__':
    main()