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
from langchain.agents import AgentExecutor, Tool, initialize_agent
from langchain.agents.agent_types import AgentType


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


def get_qa_tool(vectorstore):
    """Tool for answering questions from documents"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=ConversationBufferMemory(
            memory_key="chat_history", 
            return_messages=True,
            output_key='answer'
        ),
        return_source_documents=True,
        output_key='answer'
    )
    
    def qa_function(input_text):
        response = qa_chain({"question": input_text})
        answer = response["answer"]
        
        
        if "source_documents" in response:
            st.session_state.last_sources = [
                {
                    "source": doc.metadata['source'],
                    "page": doc.metadata['page'],
                    "content": doc.page_content[:200] + "..."  
                }
                for doc in response["source_documents"]
            ]
        
        return answer
    
    return Tool(
        name="Document_QA",
        func=qa_function,
        description="""Useful for answering specific questions that require exact information from the documents. 
        Always use this for questions that start with 'what', 'when', 'where', 'who', 'how', 'why', or 'explain'.
        Input should be a clear question."""
    )

def get_summarization_tool(vectorstore):
    """Tool for summarizing document content"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    def summarize_function(input_text):
        docs = vectorstore.similarity_search(input_text, k=3)
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""Provide a concise summary of this content relevant to: {input_text}
        
        Content:
        {combined_content}
        
        Summary:"""
        
        return llm.invoke(prompt).content
    
    return Tool(
        name="Document_Summarizer",
        func=summarize_function,
        description="""Useful when the user asks for summaries, overviews, key points, main ideas, or gist of documents.
        Also use for requests like 'TLDR', 'brief overview', or 'main takeaways'.
        Input can be a document name, topic, or general summary request."""
    )


def get_agent_executor(vectorstore):
    """Create an agent with tools for different document tasks"""
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    tools = [
        get_qa_tool(vectorstore),
        get_summarization_tool(vectorstore)
    ]
    
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        ),
        handle_parsing_errors=True
    )
    
    return agent


def display_chat_history():
    if "chat_history" in st.session_state and st.session_state.chat_history:
        with st.expander("Chat History"):
            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(f"**User:** {message.content}")
                else:
                    st.markdown(f"**Assistant:** {message.content}")
                st.write("---")

def display_sources():
    if "last_sources" in st.session_state and st.session_state.last_sources:
        with st.expander("Source Documents"):
            for source in st.session_state.last_sources:
                st.markdown(f"**Document:** {source['source']}")
                st.markdown(f"**Page:** {source['page']}")
                st.markdown(f"**Content Preview:** {source['content']}")
                st.write("---")


def handle_userinput(question):
    if not st.session_state.agent:
        st.error("Please process documents first")
        return
    
    try:
        response = st.session_state.agent({"input": question})
        
       
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        st.session_state.chat_history.extend([
            HumanMessage(content=question),
            AIMessage(content=response["output"])
        ])
        
        st.write("### Answer")
        st.write(response["output"])
        
       
        display_sources()
        display_chat_history()
        
        if "intermediate_steps" in response:
            with st.expander("Agent Thought Process"):
                for i, step in enumerate(response["intermediate_steps"]):
                    action, observation = step
                    st.write(f"**Step {i+1}**")
                    st.write(f"**Action:** {action.tool}")
                    st.write(f"**Action Input:** {action.tool_input}")
                    st.write(f"**Observation:** {observation}")
                    st.write("---")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")


def main():
    load_dotenv()
    st.set_page_config(page_title="PDF Agent", page_icon=":books:")

    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = None

    st.header("Chat with PDFs :books:")
    user_question = st.text_input("Ask about your documents:")
    
    if user_question and st.session_state.agent:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    documents = get_pdf_docs_with_metadata(pdf_docs)
                    chunks = get_text_chunks(documents)
                    vectorstore = get_vectorstore(chunks)
                    st.session_state.vectorstore = vectorstore
                    st.session_state.agent = get_agent_executor(vectorstore)
                    st.success("Ready to answer questions!")
                except Exception as e:
                    st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    from langchain.schema import HumanMessage, AIMessage
    main()