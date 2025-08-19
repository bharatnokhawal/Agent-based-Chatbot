import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate


def get_pdf_docs_with_metadata(pdf_docs):
    """Extract text along with metadata (doc name, page number)."""
    documents = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        file_name = pdf.name
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            if text: 
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": file_name, "page": i+1}
                    )
                )
    return documents



def get_text_chunks(documents):
    """Split text into chunks while preserving metadata."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks



def get_vectorstore(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    return vectorstore
    


def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    
    prompt_template = """
    You are an AI assistant helping the user answer questions based on the provided documents.
    
    Use ONLY the context below to answer the question. 
    If the context does not contain the answer, say "I could not find this information in the documents."
    
    Context:
    {context}
    
    Chat history:
    {chat_history}
    
    Question:
    {question}
    
    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "chat_history", "question"]
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}   
    )
    return conversation_chain




def handle_userinput(question):
    response = st.session_state.conversation({"question": question})
    st.session_state.chat_history = response["chat_history"]

   
    st.write("### Answer")
    st.write(response["answer"])

    
    if "source_documents" in response:
        st.write("### Sources")
        for doc in response["source_documents"]:
            st.write(
                f"📄 **{doc.metadata['source']}** - Page {doc.metadata['page']}"
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Multiple PDFs :books:")

    user_question = st.text_input("Ask a question about your documents:")

    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs here", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                
                documents = get_pdf_docs_with_metadata(pdf_docs)

               
                chunks = get_text_chunks(documents)

                
                vectorstore = get_vectorstore(chunks)

              
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
