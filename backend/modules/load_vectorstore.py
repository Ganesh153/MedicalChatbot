import os
import time
from pathlib import Path
from dotenv import load_dotenv
from tqdm.auto import tqdm # to showcase those progress bar
from pinecone import Pinecone, ServerlessSpec # to initialize pinecone instances
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
PINECONE_ENV="us-east-1"
PINECONE_INDEX_NAME="medical_index"

# Uploaded files will be stored here
UPLOAD_DIR="./uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

pc=Pinecone(api_key=PINECONE_API_KEY)
spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
existing_indexes=[i["name"] for i in pc.list_indexes()]

if PINECONE_INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=384,
        metric="dotproduct",
        spec=spec
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)

index=pc.Index(PINECONE_INDEX_NAME)

# Function for load, split, embed and upsert pdf docs content

def load_vectorstore(uploaded_files):
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    file_paths = [] # Empty list for storing the uploaded files

    # 1. Upload PDF
    for file in uploaded_files:
        save_path = Path(UPLOAD_DIR)/file.filename
        with open(save_path, "wb") as f:
            f.write(file.file.read())
        file_paths.append(str(save_path))
    
    # 2. Splitting the document
    for file_path in file_paths:
        loader=PyPDFLoader(file_path)
        documents=loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        # retrieving the text and metadata of any particular chunk
        texts = [chunk.page_content for chunk in chunks]
        metadata=[chunk.metadata for chunk in chunks]
        # getting the ids of chunks - dividing the chunks with the name of id
        ids = [f"{Path(file_path).stem} - {i}" for i in range(chunks)]

        # 3. Embedding
        print("Embedding chunks")
        # embedded vectors are stored in this embedding variable
        embedding = embed_model.embed_documents(texts)

        # 4. Upsert these to Pinecone vector database
        print("Upserting embeddings to pinecone vector database")
        with tqdm(total=len(embedding), desc="Upserting to Pinecone DB") as progress:
            index.upsert(vectors=zip(ids, embedding, metadata))
            progress.update(len(embedding))

        print(f"Upload complete for {file_path}")





