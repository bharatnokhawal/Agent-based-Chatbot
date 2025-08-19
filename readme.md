# 📚 PDF Conversational Agent

An **AI-powered Streamlit app** that allows you to upload multiple PDFs and **chat with them** using Google Gemini.  
The app supports **question answering, summarization, and conversation history**, and shows **source documents** for transparency.  

---

## 🚀 Features
- 📄 Upload multiple PDF documents  
- 🔎 Ask **questions** and get accurate answers from the documents  
- 📝 Get **summaries, key points, and overviews**  
- 🧠 Maintains **chat history** (context-aware conversations)  
- 📌 Shows **source document, page number, and content preview** for each answer  
- 🤖 Built with **LangChain + Gemini + FAISS + Streamlit**  

---

## 📂 Project Structure
├── app.py # Main Streamlit application
├── requirements.txt # Project dependencies
└── README.md # Project documentation

yaml
Copy
Edit

---

## ⚙️ Installation

1. **Clone this repo**
   ```bash
   git clone https://github.com/your-username/pdf-agent.git
   cd pdf-agent
Create virtual environment (recommended)

bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # On Mac/Linux
venv\Scripts\activate      # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables

Create a .env file in the project root

Add your Google API Key:

env
Copy
Edit
GOOGLE_API_KEY=your_api_key_here
▶️ Run the App
bash
Copy
Edit
streamlit run app.py
Then open the local URL (usually http://localhost:8501) in your browser.

🛠️ Tech Stack
Streamlit – Web UI

LangChain – LLM framework

Google Gemini – LLM & Embeddings

FAISS – Vector database for document search

PyPDF2 – PDF text extraction

📖 Usage Guide
Upload one or more PDF files from the sidebar

Click Process to index the documents

Type your question in the input box (e.g., "Summarize chapter 2" or "Who is the author?")

Get an AI-generated response with source references

📌 Example Queries
"Summarize this document in 5 bullet points"

"What are the main findings in section 3?"

"Who is mentioned as the key contributor?"

🖼️ Architecture
scss
Copy
Edit
 PDF(s) → Text Extraction (PyPDF2) 
        → Text Splitting (LangChain) 
        → Embeddings (Google Generative AI) 
        → FAISS Vector Store 
        → Agent with Tools (Q&A + Summarization) 
        → Streamlit UI
🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.
