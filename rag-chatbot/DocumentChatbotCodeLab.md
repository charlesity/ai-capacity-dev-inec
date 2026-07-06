# Build a PDF-based Chatbot with PyTorch

## Learning Objectives

By the end of this codelab, participants will be able to:

- Create and activate a Python virtual environment.
- Install project dependencies using `requirements.txt`.
- Build a Streamlit application from scratch.
- Load PDF documents using LangChain.
- Split large documents into overlapping chunks.
- Generate semantic embeddings with Sentence Transformers.
- Retrieve relevant document chunks using cosine similarity.
- Use a Hugging Face BERT (Bidirectional Encoder Representations from Transformers) Question Answering model to extract answers.
- Manage chat history using Streamlit session state.
- Explain the complete Retrieval + Reader architecture.

## Best Practices

### Project Organization
- Keep source code in `app.py` and PDFs in a dedicated `data/` folder.
- Store dependencies in `requirements.txt`.
- Use descriptive function names.

### Model Loading
- Cache pretrained models with `@st.cache_resource`.
- Download models once and reuse them.

### Document Processing
- Split long documents into overlapping chunks.
- Experiment with chunk sizes (500–1000 characters) for this use case.

### Performance
- Wrap inference in `torch.no_grad()` to reduce memory usage.
- Cache embeddings when documents rarely change.

### Security
- Never hard-code API keys or secrets.
- Add large model folders to `.gitignore` if needed.

### User Experience
- Display progress with `st.spinner()`.
- Show informative messages when no PDFs are available.
- Preserve conversation history with `st.session_state`.

### Testing
- Test with multiple PDF types.
- Verify answers against the original documents.

---

The remainder of this codelab follows these steps:
1. Create a virtual environment.
2. Install dependencies.
3. Create the project structure.
4. Import libraries.
5. Configure Streamlit.
6. Load pretrained models.
7. Process documents into embeddings.
8. Build the sidebar.
9. Build the chat interface.
10. Retrieve relevant chunks.
11. Run BERT Question Answering.
12. Display answers.
13. Execute the complete chat loop.
14. Run the application with `streamlit run app.py`.


---

# Step 1. Create a Virtual Environment

## Why?

A virtual environment isolates your project's dependencies.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

# Step 2. Install Dependencies

Create **requirements.txt**

```text
streamlit
torch
transformers
sentence-transformers
langchain-community
langchain-text-splitters
pypdf
```

Install packages

```bash
pip install -r requirements.txt
```

---

# Step 3. Create the Project Structure

```text
torch-chatbot/
│
├── app.py
├── requirements.txt
└── data/
    ├── document1.pdf
    └── document2.pdf
```

---

## Step 4. Import Libraries

### Explanation
This section imports all the libraries required for the application. Streamlit builds the web interface, LangChain loads and splits PDF documents, Sentence Transformers creates semantic embeddings, Hugging Face Transformers provides the BERT Question Answering model, and PyTorch runs the neural network inference.

```python
import os
import streamlit as st
import torch
import torch.nn as nn
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer, util
```

## Step 5. Configure Streamlit

### Explanation
This code configures the Streamlit page, creates the application title and description, and ensures a `data` directory exists to store PDF documents.

```python
st.set_page_config(page_title="Torch Chatbot", page_icon="", layout="wide")
st.title("Chat with a document")
st.write("Scan and extract answers from local PDFs.")

DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)
```

## Step 6. Load Pretrained Models

### Explanation
This function downloads (the first time) and caches two AI models: a Sentence Transformer for retrieving relevant document chunks and a BERT Question Answering model for extracting answers. The `@st.cache_resource` decorator prevents the models from being reloaded every time Streamlit refreshes.

```python
@st.cache_resource(show_spinner="Loading PyTorch models")
def load_pytorch_models():
    retriever_model = SentenceTransformer("all-MiniLM-L6-v2")
    reader_model_name = "deepset/bert-large-uncased-whole-word-masking-squad2"
    tokenizer = AutoTokenizer.from_pretrained(reader_model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(reader_model_name)
    return retriever_model, tokenizer, model

retriever_model, tokenizer, reader_model = load_pytorch_models()
```

## Step 7. Process Documents into Embeddings

### Explanation
This function loads every PDF, splits them into smaller overlapping chunks, and converts each chunk into a numerical embedding. These embeddings are later compared with the user's question to find the most relevant document sections.
The user's question is converted into an embedding. Cosine similarity compares this embedding with all document embeddings, and the top three most relevant chunks are selected as context for question answering.
```python
@st.cache_resource(show_spinner="Vectorizing local PDFs")
def process_documents():
    loader = PyPDFDirectoryLoader(DATA_DIR)
    docs = loader.load()

    if not docs:
        return None, None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)
    texts = [doc.page_content for doc in splits]

    with torch.no_grad():
        embeddings = retriever_model.encode(texts, convert_to_tensor=True)

    return texts, embeddings


texts, doc_embeddings = process_documents()
```

## Step 8. Build the Sidebar

### Explanation
The sidebar lists every PDF found in the `data` directory. This allows users to verify which documents have been loaded into the application.

```python
with st.sidebar:
    st.header("File(s) in `/data`")
    pdf_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.pdf')]
    if pdf_files:
        st.success(f"Cached {len(pdf_files)} PDF file(s).")
        for file in pdf_files:
            st.markdown(f"- `{file}`")
    else:
        st.warning("No PDFs found! Drop your PDFs in the `/data` directory to start.")

```

## Step 9. Build the Chat Interface

### Explanation
This section initializes the chat history using Streamlit's session state and redisplays previous messages whenever the application reruns.

```python
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```

## Step 10. Retrieve Relevant Chunks

### Explanation


```python
with torch.no_grad():
    ...
```

## Step 11. Run BERT Question Answering

### Explanation
The tokenizer prepares both the question and the retrieved context for the BERT model. The model predicts where the answer starts and ends within the context, and those tokens are decoded into readable text.

```python
inputs = tokenizer(...)
```

## Step 12. Display Answers

### Explanation
If a valid answer span is found, it is displayed to the user. Otherwise, a friendly message indicates that no answer could be extracted.

```python
if extracted_string.strip() and start_idx < end_idx:
    ...
```

## Step 13. Execute the Complete Chat Loop

### Explanation
This final section combines all the previous steps into a working chatbot. It accepts the user's question, retrieves relevant document chunks, performs BERT question answering, displays the answer, and stores the conversation history.

```python
# Complete chat loop from the application
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if texts is None:
    st.info("Please add PDF documents to your local `data/` directory and refresh the app.")
else:
    if user_query := st.chat_input("Ask a factual question about your documents..."):

        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.messages.append({"role": "user", "content": user_query})

        with st.chat_message("assistant"):
            with st.spinner("Searching and parsing result chunks"):

                with torch.no_grad():
                    query_embedding = retriever_model.encode(user_query, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
                    top_results = torch.topk(cos_scores, k=min(3, len(texts)))

                context_chunks = [texts[idx] for idx in top_results.indices.tolist()]
                combined_context = "\n\n".join(context_chunks)

                inputs = tokenizer(
                    user_query,
                    combined_context,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )

                with torch.no_grad():
                    outputs = reader_model(**inputs)

                start_idx = torch.argmax(outputs.start_logits)
                end_idx = torch.argmax(outputs.end_logits)

                answer_tokens = inputs.input_ids[0][start_idx: end_idx + 1]
                extracted_string = tokenizer.decode(answer_tokens, skip_special_tokens=True)

                if extracted_string.strip() and start_idx < end_idx:
                    answer = f"**Answer extracted from source:** \n> {extracted_string.strip()}"
                else:
                    answer = "I couldn't locate a definitive answer snippet matching your query inside the context document blocks."

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
```

## Step 14. Run the Application
Start the Streamlit development server using the following command. Open the local URL displayed in the terminal, place your PDF documents into the `data` folder, refresh the application, and begin asking questions.

```bash
streamlit run app.py
```

Open the URL shown in the terminal, add PDF files to the `data` folder, refresh the page, and begin asking questions.

## Congratulations!

You have built a Retrieval + Reader document question-answering application using:
- Streamlit
- LangChain PDF Loader
- Sentence Transformers
- Hugging Face BERT QA
- PyTorch

### Exercise
- Add a new PDF file to the `data` folder.
- Ask a question about the new PDF.
- Refresh the page and ask a new question.
