import streamlit as st
import pickle
import os
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmplTemplates import css,bot_template,user_template
from langchain.callbacks import get_openai_callback


# ===================Function=================================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader =PdfReader(pdf)
        for page in pdf_reader.pages:
           text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks, pdf_docs):
    embeddings = OpenAIEmbeddings()

    # if pdf_docs is not None:
    for pdf in pdf_docs:
        store_name = pdf.name[:-4]
        if os.path.exists(f"./store/{store_name}"):
            vectorstore = FAISS.load_local(f"./store/{store_name}", embeddings)
            st.write("You Already upload this pdf")
            # st.write(store_name)
        else:
            vectorstore = FAISS.from_texts(texts = text_chunks, embedding=embeddings)
            vectorstore.save_local(f"./store/{store_name}")
            # st.write(store_name)
    # else:
    #     store_name = vectorstore_path
    #     vectorstore = FAISS.load_local(f"./store{store_name}", embeddings)
    
    return vectorstore

# def get_vectorstore_db(directory="./store"):
#     embeddings = OpenAIEmbeddings()
#     vectorstores = []

#     for filename in os.listdir(directory):
#         if filename.endswith(".faiss"):
#             store_name = os.path.join(directory, filename)
#             vectorstore = FAISS.load_local(store_name, embeddings)
#             vectorstores.append(vectorstore)

#     return vectorstores

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history',return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question': user_question})
        print(cb)
    st.session_state.chat_history = response['chat_history']

    # Display the current user question
    st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)

    # Display the bot's response
    st.write(bot_template.replace("{{MSG}}", response['chat_history'][-1].content), unsafe_allow_html=True)


#===================frontend================================
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with project", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = None
    
    st.header("chat with project")
    user_question = st.text_input("Ask a question about your Docs:")
    if user_question:
        handle_userinput(user_question)

   
    with st.sidebar:
        st.subheader("Your Docs")
        pdf_docs = st.file_uploader(
            "Upload your PDFs", accept_multiple_files=True, type='pdf')
        
        # vectorstore = get_vectorstore_db()
        # st.session_state.conversation = get_conversation_chain(vectorstore)
  
        # if pdf_docs is not None:
        if st.button("Process"):
            with st.spinner("Processing"):
                #Get pdf text        
                raw_text = get_pdf_text(pdf_docs)
                

                #get text chunks
                text_chunks = get_text_chunks(raw_text)
                
        
                #create vector store
                vectorstore = get_vectorstore(text_chunks,pdf_docs)
                
                #create conversation
                st.session_state.conversation = get_conversation_chain(vectorstore)
    



if __name__ == '__main__':
    main()

