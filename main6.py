from datetime import datetime
import streamlit as st
import os
import pdfplumber
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import Document
import time
from gtts import gTTS
import base64
from dotenv import load_dotenv


# Define a simple Document class if not available
class Document:
    def __init__(self, page_content):
        self.page_content = page_content


# Load environment variables
load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Use Hugging Face embeddings
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create LLM model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192", max_tokens=1024)

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question from the research_papers
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Load PDF documents using pdfplumber
        documents = []
        pdf_directory = "research_papers"
        for filename in os.listdir(pdf_directory):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(pdf_directory, filename)
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            documents.append(Document(page_content=text))

        st.session_state.docs = documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("😎 My Research Assistant")

# Check if the vector database needs to be created
if "vectors" not in st.session_state:
    create_vector_embedding()
    st.write("Vector Database is ready")

user_prompt = st.text_input("Ask your questions related to Drupal, AWS, Moodle or Generative AI")

# Process the user prompt and generate a response
if user_prompt:
    timestamp1 = datetime.now().strftime("%d-%m-%Y")
    timestamp2 = datetime.now().strftime("%H:%M:%S")
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retrieval_chain.invoke({'input': user_prompt})
    response_time = time.process_time() - start_time

    st.session_state.chat_history.append({
        "user": user_prompt,
        "bot": response['answer'],
        "timestamp2": timestamp2,
        "timestamp1": timestamp1
    })

    st.text_area("Response", value=response['answer'], height=200)
    st.write(f"Responded in: {response_time:.2f} seconds")
    user_prompt=st.empty()

    st.markdown("<h6>Listen to the response here:</h6>", unsafe_allow_html=True)
    audio_placeholder = st.empty()

    tts = gTTS(text=response['answer'], lang='en')
    audio_file_path = "response.mp3"
    tts.save(audio_file_path)

    with open(audio_file_path, "rb") as audio_file:
        audio_b64 = base64.b64encode(audio_file.read()).decode()
        audio_html = f"""
        <audio controls>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
            Your browser does not support the audio element.
        </audio>
        """
        audio_placeholder.markdown(audio_html, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("<h2 style='text-decoration: underline; color:green;'>Chat History</h2>", unsafe_allow_html=True)
        for chat in reversed(st.session_state.chat_history):
            st.write(f"**Asked at**: {chat['timestamp2']}, {chat['timestamp1']}")
            st.write(f"**You**: {chat['user']}")
            st.write(f"**Assistant**: {chat['bot']}")
            st.write("---")

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')

# Custom CSS to pin text to the bottom
footer = """
<style>
    .footer {
        position: sticky;
        bottom: 0;
        left: 0;
        width: 100%;
        font-size: 8px;
        color: grey;
        text-align: center;
    }
</style>
<div class="footer">
<br>
<br>
<br>
<br>
    <p>Created using Groq, Hugging Face, FAISS & Langchain</p>
</div>
"""

st.write(footer, unsafe_allow_html=True)
