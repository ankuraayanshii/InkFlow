# Multi-PDF Chat Agent 🤖

This Streamlit application allows users to upload multiple PDF documents, extract text, and then interact with the content through a conversational AI interface. The application uses Google Generative AI for generating embeddings and FAISS for efficient similarity search.

## Features

- **PDF Upload**: Users can upload multiple PDF files.
- **Text Extraction**: Extracts text from the uploaded PDF files.
- **Text Indexing**: Indexes extracted text using FAISS for efficient searching.
- **Conversational AI**: Allows users to ask questions and get answers based on the content of the uploaded PDFs.

## 🎯 How It Works:

The application follows these steps to provide responses to your questions:

1. **PDF Loading**: The app reads multiple PDF documents and extracts their text content.
2. **Text Chunking**: The extracted text is divided into smaller chunks that can be processed effectively.
3. **Language Model**: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. **Similarity Matching**: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. **Response Generation**: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## 🎯 Key Features

- **Adaptive Chunking**: Our Sliding Window Chunking technique dynamically adjusts window size and position for RAG, balancing fine-grained and coarse-grained data access based on data complexity and context.
- **Multi-Document Conversational QA**: Supports simple and multi-hop queries across multiple documents simultaneously, breaking the single-document limitation.
- **File Compatibility**: Supports both PDF and TXT file formats.
- **LLM Model Compatibility**: Supports Google Gemini Pro, OpenAI GPT 3, Anthropic Claude, Llama2 and other open-source LLMs.

## 🌟Requirements

- Streamlit
- google-generativeai
- python-dotenv
- langchain
- PyPDF2
- faiss-cpu
- langchain_google_genai


## Prerequisites

Before you can run this application, you need to have the following installed:
- Python 3.8 or higher
- pip (Python package installer)

## Installation

Follow these steps to set up the project environment:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/multi-pdf-chat-agent.git
   cd multi-pdf-chat-agent
Install Dependencies

copy
pip install -r requirements.txt
Set Up Environment Variables

Create a .env file in the root directory of the project.
Set up your Google API key from https://makersuite.google.com/app/apikey by creating a .env file in the root directory of the project with the following contents:

GOOGLE_API_KEY=your_google_api_key_here
Running the Application
To run the application, execute the following command in the terminal:

copy
streamlit run app.py
Navigate to http://localhost:8501 in your web browser to view and interact with the application.

Usage
Upload PDFs: Use the sidebar to upload one or more PDF files.
Process PDFs: Click the "Process PDFs" button to extract and index the text.
Ask Questions: Enter your question in the input field to get answers based on the uploaded PDF content.
Contributing
Contributions to this project are welcome! Here are a few ways you can help:

Report bugs and request features by creating issues.
Improve documentation.
Submit pull requests with bug fixes or new features.


Acknowledgments
Thanks to Google Generative AI for the AI models.
Thanks to the developers of FAISS for providing efficient similarity search algorithms.
Contact
For any additional questions or feedback, please contact ankurdmps123@gmaul.com.

Made with ❤️ Ankur#   I n k F l o w  
 