import os
import chromadb
import PyPDF2
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini API
api_key = os.environ.get("GOOGLE_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

class GeminiEmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        # Use genai.embed_content to get embeddings
        response = genai.embed_content(
        model="models/gemini-embedding-001", # <-- THIS IS THE ONLY CHANGE
        content=input,
        task_type="retrieval_document"
    )
        return response['embedding']

def build_or_load_db():
    client = chromadb.PersistentClient(path="./chroma_db")
    embedding_function = GeminiEmbeddingFunction()
    collection = client.get_or_create_collection(
        name="real_estate_knowledge",
        embedding_function=embedding_function
    )

    kb_path = "knowledge_base"
    if not os.path.exists(kb_path):
        os.makedirs(kb_path)

    pdf_files = [f for f in os.listdir(kb_path) if f.endswith(".pdf")]
    
    if pdf_files and collection.count() == 0:
        for pdf_file in pdf_files:
            file_path = os.path.join(kb_path, pdf_file)
            with open(file_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                
                # Split text into chunks of 500 characters
                chunks = [text[i:i+500] for i in range(0, len(text), 500)]
                
                # Add to collection
                ids = [f"{pdf_file}_{i}" for i in range(len(chunks))]
                collection.add(
                    documents=chunks,
                    ids=ids
                )
    return collection

def retrieve_context(query):
    collection = build_or_load_db()
    results = collection.query(
        query_texts=[query],
        n_results=3
    )
    # results['documents'] is a list of lists
    context_chunks = results['documents'][0] if results['documents'] else []
    return "\n\n".join(context_chunks)
