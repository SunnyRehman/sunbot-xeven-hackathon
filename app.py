################################# IMPORTS ###############################################
import streamlit as st
import os
import pdfplumber
import docx2txt
from zipfile import ZipFile
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from streamlit_chat import message
import pinecone
from langchain.vectorstores import Pinecone
from langchain.schema import (SystemMessage,HumanMessage,AIMessage)


####################### INITIAL CONFIGURATIONS - GETTING APIS FROM TOML ####################### 

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets['PINECONE_API_KEY']
PINECONE_ENV = st.secrets['PINECONE_ENV']


####################################### MAIN FUNCTION #####################################

# "with" notation
def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with your file")
    st.header("🌞 Welcome to SunBot - Ask Your Data")

    ####################################### SESSION VARIABLES #####################################

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    ####################################### SIDE BAR #####################################

    with st.sidebar:
        
        st.image("https://sunskilltech.com/portal/sunbot-600.jpg",width=120)
        st.header('🌞 SunBot')
        uploaded_files = st.file_uploader("Please Upload your file", type=['pdf', 'docx', 'zip'], accept_multiple_files=True)
        OPENAI_API = OPENAI_API_KEY
        process = st.button("Process")
        
    ####################################### JAB USER PROCESS KO CLICK KARE   #########################################
    if process:
        # if not uploaded_files:
        #     st.error("Please upload at least one file to proceed.")
        #     st.stop()

        if not OPENAI_API:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        #AGAR KEY HAI TOU JO FILES UPLOAD HUI HAIN UNPE GET_FILES_TEXT KA FUNCTION LAGAO
        files_text = get_files_text(uploaded_files)
        st.write("File Loaded...")

        # st.write(files_text)

    #################################### CHUNKS ###############################
        text_chunks_character = get_chunks_by_character(files_text)
        st.write("File Chunks Created - Split By Character..")
        # st.write(text_chunks)

        text_chunks_recursive = recursively_split_by_character(files_text)
        st.write("File Chunks Created - Recursively Split By Character..")
        # st.write(text_chunks_character)


    #################################### VECTOR STORE ###############################

        #create vetore stores
        vetorestore = get_vectorstore(text_chunks_character)
        st.write("Vector Store Created...")

        st.write("All Done! Have Fun!...")

        
        
        # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore,OPENAI_API_KEY)


        st.session_state.processComplete = True

    if st.session_state.processComplete == True:
        user_question = st.chat_input("Ask a Question about your data.")
        if user_question:
            handel_userinput(user_question)

        
        ####################################################################


####################################################### File Loading ##################################################################

#1. Custom Loader for Zip Files
#2. PDF Loader using PDF Plumber
#3  DOCX Loader using Docx2Txt

# Custom Loader to handle Zip Files
def process_zip_files(uploaded_zip):
    text = ""
    with ZipFile(uploaded_zip, 'r') as zip_file:
        for file_info in zip_file.infolist():
            # Extract each file in the zip archive
            with zip_file.open(file_info) as extracted_file:
                file_extension = os.path.splitext(file_info.filename)[1].lower()

                if file_extension == ".pdf":
                    text += get_pdf_text(extracted_file)
                elif file_extension == ".docx":
                    text += get_docx_text(extracted_file)
                # Add other file types if needed
    return text

################ GET INPUT & READ TEXT  ##########################

def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1].lower()

        if file_extension == ".zip":
            text += process_zip_files(uploaded_file)
            
            # st.write(f"File Name: {uploaded_file.name}")
            # st.write("File Text:", text)
        
        elif file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text

################ PDF LOADER - USING PDF PLUMBER ##########################

def get_pdf_text(pdf):
    text = ""
    with pdfplumber.open(pdf) as pdf_document:
        for page in pdf_document.pages:
            text += page.extract_text()
    return text

################ DOCX LOADER - USING Docx2Txt #############################

def get_docx_text(file):
    text = docx2txt.process(file)
    return text

################ EMPTY #############################

def get_csv_text(file):
    return "a"


############################# DOCUMENT SPLITTING #############################

################################# SPLIT BY CHARACTER ########################

def get_chunks_by_character(text):
    # SPLITTING INTO CHUNKS
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    return text_splitter.create_documents(chunks)

############################ RECURSIVELY SPLIT BY CHARACTER ##########################

def recursively_split_by_character(text, chunk_size=1000, chunk_overlap=200, min_chunk_size=50):
    if len(text) <= chunk_size:
        return [text]

    # Split the text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)

    # Recursively split each chunk
    result = []
    for chunk in chunks:
        result.extend(recursively_split_by_character(chunk, chunk_size, chunk_overlap, min_chunk_size))

    # Filter out small chunks
    result = [chunk for chunk in result if len(chunk) >= min_chunk_size]

    return result



############################# EMBEDDINGS #############################

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    embeddings_2 = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    knowledge_base = Pinecone.from_documents(
        documents=text_chunks,  # Pass the text chunks as the 'documents' argument
        embedding=embeddings,  # Provide the embeddings object
        index_name='hackathon-sunny'
    )

    return knowledge_base


############################# GUARDRAILS ###############################################


def build_message_list():
    """
    Build a list of messages including system, human and AI messages.
    """
    # Start zipped_messages with the SystemMessage
    zipped_messages = [SystemMessage(
        # content="You are a helpful AI assistant talking with a human. If you do not know an answer, just say 'I don't know', do not make up an answer.")]
        content = """You are an AI chat bot that answers users according to data provided in the uploaded documents or in vector database. Please provide accurate and helpful information, and always maintain a polite and professional tone.

                1. Greet the user politely and always be respectful.
                2. Do not provide offensive, inappropriate, sensitive or harmful content. Refrain from engaging in any form of discrimination, harassment, or inappropriate behavior.
                3. Be patient and considerate when responding to user queries, and provide clear explanations.
                4. If the user expresses gratitude or indicates the end of the conversation, respond with a polite farewell.
                5. Do Not generate the long paragarphs in response. Maximum Words should be 100.

                Remember, your primary goal is to answer questions from files and data available to you"""
    )]


    # Zip together the past and generated messages
    for human_msg, ai_msg in zip_longest(st.session_state.conversation, st.session_state.chat_history):
        if human_msg is not None:
            zipped_messages.append(HumanMessage(
                content=human_msg))  # Add user messages
        if ai_msg is not None:
            zipped_messages.append(
                AIMessage(content=ai_msg))  # Add AI messages

    return zipped_messages


############################# CONVERSATION CHAIN : 10 MARKS #############################


def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-3.5-turbo-16k',temperature=0)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vetorestore.as_retriever(),
        memory=memory
    )
    return conversation_chain


############################ USER INPUT FOR PROMPT ##########################

def handel_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({'question':user_question})
    st.session_state.chat_history = response['chat_history']

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))

if __name__ == '__main__':
    main()