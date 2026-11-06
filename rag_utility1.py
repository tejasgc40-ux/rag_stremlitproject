import os
import base64
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA


# --------------------------------------------------
# Load environment variables from the project's .env
# --------------------------------------------------

env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

working_dir = str(Path(__file__).resolve().parent)


# --------------------------------------------------
# Load embedding model
# --------------------------------------------------

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"
)


# --------------------------------------------------
# Initialize Groq LLM for PDF question answering
# --------------------------------------------------

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# --------------------------------------------------
# Process uploaded PDF and store chunks in Chroma
# --------------------------------------------------

def process_document_to_chroma_db(file_name):
    """
    Load a PDF, split it into chunks, and store the chunks
    in a persistent Chroma vector database.
    """

    pdf_path = Path(working_dir) / file_name

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    loader = UnstructuredPDFLoader(str(pdf_path))
    documents = loader.load()

    if not documents:
        raise ValueError("No readable text was found in the PDF.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )

    texts = text_splitter.split_documents(documents)

    if not texts:
        raise ValueError("The PDF did not produce any text chunks.")

    vectorstore_path = Path(working_dir) / "doc_vectorstore"

    Chroma.from_documents(
        documents=texts,
        embedding=embedding,
        persist_directory=str(vectorstore_path)
    )

    return "PDF processed successfully."


# --------------------------------------------------
# Answer questions using the uploaded PDF
# --------------------------------------------------

def answer_question(user_question):
    """
    Retrieve relevant PDF content from Chroma and answer
    the user's question using the retrieved context.
    """

    vectorstore_path = Path(working_dir) / "doc_vectorstore"

    if not vectorstore_path.exists():
        return "No processed PDF database found. Please upload and process a PDF first."

    # Load the existing Chroma database
    vectordb = Chroma(
        persist_directory=str(vectorstore_path),
        embedding_function=embedding
    )

    # Retrieve relevant PDF chunks
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 4}
    )

    retrieved_docs = retriever.invoke(user_question)

    # Check whether Chroma found any relevant content
    if not retrieved_docs:
        return (
            "I couldn't find relevant text in the processed PDF. "
            "Please check whether the PDF contains readable text, "
            "then process it again."
        )

    # Combine retrieved PDF text into context
    context = "\n\n".join(
        document.page_content for document in retrieved_docs
    )

    # Prompt instructs the model to answer from the PDF context
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful PDF question-answering assistant.

Answer the user's question using only the information in the PDF context below.

If the answer is not present in the context, say:
"I couldn't find this information in the uploaded PDF."

Do not claim that you cannot access the PDF if context is provided.

PDF Context:
{context}

User Question:
{question}

Answer:
"""
    )

    # Generate answer using retrieved PDF context
    formatted_prompt = prompt.format(
        context=context,
        question=user_question
    )

    response = llm.invoke(formatted_prompt)

    return response.content     

# --------------------------------------------------
# Answer questions about an uploaded image using Groq
# --------------------------------------------------

def answer_image_question(uploaded_image, user_question):
    """
    Send an uploaded image and question to Groq's vision model.
    """

    if uploaded_image is None:
        return "Please upload an image first."

    if not user_question or not user_question.strip():
        return "Please enter a question about the image."

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY was not found. Check your project's .env file."
        )

    image_bytes = uploaded_image.getvalue()
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime_type = uploaded_image.type

    client = Groq(api_key=api_key)

    response = client.chat.completions.create(
       model="qwen/qwen3.8-27b", 
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        temperature=0.2,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content