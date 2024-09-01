Multi-PDF Chat Agent 🤖
The Multi-PDF Chat Agent is a Streamlit application that allows users to upload multiple PDF documents, extract text, and interact with the content through a conversational AI interface. The application leverages Google Generative AI for generating embeddings and FAISS for efficient similarity search.

Features
PDF Upload: Users can upload multiple PDF files.

Text Extraction: Extracts text from the uploaded PDF files.

Text Indexing: Indexes extracted text using FAISS for efficient searching.

Conversational AI: Allows users to ask questions and receive answers based on the content of the uploaded PDFs.

How It Works
PDF Loading: The app reads multiple PDF documents and extracts their text content.

Text Chunking: The extracted text is divided into smaller chunks for effective processing.

Language Model: The application generates vector representations (embeddings) of the text chunks using a language model.

Similarity Matching: When a user asks a question, the app compares it with the text chunks and identifies semantically similar ones.

Response Generation: Relevant chunks are passed to the language model, which generates a response based on the PDF content.

Key Features
Adaptive Chunking: Sliding Window Chunking dynamically adjusts window size and position for efficient data access.

Multi-Document Conversational QA: Supports queries across multiple documents, breaking the single-document limitation.

File Compatibility: Handles both PDF and TXT file formats.

LLM Model Compatibility: Works with various language models, including Google Gemini Pro, OpenAI GPT-3, Anthropic Claude, and Llama2.
Requirements

Make sure you have the following installed:

Python 3.8 or higher
pip (Python package installer)
Installation
Clone the Repository
git clone https://github.com/ankuraayanshii/InkFlow.git
cd multi-pdf-chat-agent

Install Dependencies
pip install -r requirements.txt

Set Up Environment Variables
Create a .env file in the root directory of the project.
Set your Google API key:
GOOGLE_API_KEY=your_google_api_key_here

Running the Application
Execute the following command in the terminal:

streamlit run chatapp.py

Navigate to http://localhost:8501 in your web browser to view and interact with the application.

Usage
Upload PDFs: Use the sidebar to upload one or more PDF files.
Process PDFs: Click the “Process PDFs” button to extract and index the text.
Ask Questions: Enter your question in the input field to get answers based on the uploaded PDF content.
Contributing
Contributions to this project are welcome! You can:

Report bugs and request features by creating issues.
Improve documentation.
Submit pull requests with bug fixes or new features.
Acknowledgments
Thanks to Google Generative AI for the AI models.
Thanks to the developers of FAISS for providing efficient similarity search algorithms.
Contact
For questions or feedback, please contact ankurdmps123@gmail.com.

Made with ❤️ by Ankur
