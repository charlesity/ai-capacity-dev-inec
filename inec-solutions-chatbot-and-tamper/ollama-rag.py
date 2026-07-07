import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

st.set_page_config(page_title="Ollama RAG Chatbot", page_icon="", layout="wide")
st.image("path/to/your/banner_image.png", use_container_width=True)

st.title("Local Ollama RAG Chatbot")
st.write("Ask questions about your documents using a 100% local LLM pipeline.")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)


# Cache the RAG initialization so your PDFs are processed only once
@st.cache_resource(show_spinner="Indexing your local PDF files...")
def build_ollama_rag():
    # 1. Load PDFs
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()
    if not docs:
        return None

    # 2. Chunk text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # 3. Create Local Embeddings (Runs entirely on your machine)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # 4. Spin up local Vector Database
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 5. Define the Prompt
    system_prompt = (
        "You are a helpful organizational assistant for (Independent National Electoral Commission) INEC Nigeria. Use the following context fragments "
        "to answer the question thoroughly but concisely. If you do not know the answer "
        "based on the context, state clearly that the information is missing from the documents.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 6. Initialize local Ollama Model
    # Make sure 'llama3.2' matches the model name you pulled in Ollama
    llm = OllamaLLM(model="llama3.2", temperature=0.3)

    # Link everything together
    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)


# --- Sidebar ---
with st.sidebar:
    st.header("Indexed Documents")
    files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if files:
        st.success(f"Found {len(files)} local PDF file(s).")
        for f in files:
            st.markdown(f"- `{f}`")
    else:
        st.warning("Please add PDF documents into your local `/data` folder to start.")

# Initialize the chain
rag_chain = build_ollama_rag()

# Manage Session State Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display ongoing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Functionality ---
if rag_chain is None:
    st.info("The application is ready. Drop some PDFs into the `data/` folder and refresh your page to start chatting.")
else:
    if user_query := st.chat_input("Ask a question about your documents..."):
        # Show User Input
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        # Generate Local Ollama Output
        with st.chat_message("assistant"):
            with st.spinner("Ollama is processing chunks and thinking..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(
                        f"Failed to communicate with Ollama: {e}. Check if `ollama run` is running in your terminal.")