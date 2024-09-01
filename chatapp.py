import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or " "  # Ensure there's a fallback for empty pages
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.session_state.chat_history.append({"question": user_question, "answer": response["output_text"]})
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Multi PDF Chatbot", page_icon=":scroll:", layout="wide")
    st.header("📚 Multi-PDF Chat Agent 🤖")

    with st.sidebar:
        st.image("img/Robot.jpg", width=300)
        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded:")
    if user_question:
        user_input(user_question)

    if 'chat_history' in st.session_state:
        for chat in st.session_state.chat_history:
            st.text_area("Question", chat['question'], height=75)
            st.text_area("Answer", chat['answer'], height=150)

    st.markdown("""
    <div style="background-color: #0E1117; color: white; padding: 10px; text-align: center; position: fixed; bottom: 0; width: 100%;">
        Made with ❤️ by <a href="https://github.com/ankuraayanshii" style="color: #FF4B4B; text-decoration: none;">Ankur</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
