
from langchain.vectorstores import Milvus
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

from .deepseek_engine import deepseek_model
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pymilvus import Collection, connections, FieldSchema, CollectionSchema, DataType, utility
from openai import OpenAI
from pymilvus import MilvusException,MilvusClient
import tiktoken
from io import BytesIO
import pdfplumber
from bs4 import BeautifulSoup
import docx
import os
from fastapi import FastAPI, UploadFile, File
# from exceptions import PendingDeprecationWarning


from sentence_transformers import SentenceTransformer

# Manually load the model
model = SentenceTransformer('all-MiniLM-L6-v2')


local_model = SentenceTransformer("all-MiniLM-L6-v2")
openai_api_key =""       # Use environment variable or config management

client = OpenAI(api_key=openai_api_key)
# Connect to Milvus vector store



milvus_client = MilvusClient(
    uri = "http://3.238.28.60:19530", # Dev server
    token="root:Milvus"
)


# Helper functions

def count_tokens(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def extract_text_from_pdf(pdf_file: BytesIO):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(docx_file: BytesIO):
    doc = docx.Document(docx_file)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_html(html_file: BytesIO):
    soup = BeautifulSoup(html_file, "html.parser")
    return soup.get_text()

def extract_content(file: UploadFile):
    file_content = file.file.read()
    if file.filename.endswith(".pdf"):
        return extract_text_from_pdf(BytesIO(file_content))
    elif file.filename.endswith(".docx"):
        return extract_text_from_docx(BytesIO(file_content))
    elif file.filename.endswith(".html"):
        return extract_text_from_html(BytesIO(file_content))
    else:
        raise ValueError("Unsupported document format")



def get_hybrid_embedding(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    text_content = text_splitter.split_text(text)
    print(">>>>>>>.",text_content)
    response = client.embeddings.create(input=text_content, model="text-embedding-3-large")
    return response.data[0].embedding



def create_document_entry(text_chunk, category, filename):
    return {
        "text": text_chunk,
        "embedding": get_hybrid_embedding(text_chunk),
        "category": category,
        "filename": filename,
    }

def get_file_metadata(file: UploadFile):
    return {
       "file_name": os.path.basename(file.filename),
    }

def split_text(text, max_length=1500):
    chunk_overlap = 250
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_length, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return [chunk[:max_length] for chunk in chunks]

def create_or_get_milvus_lite_collection():
    collection_name = "Global_Regulatory"

    if collection_name in utility.list_collections():
        collection = Collection(name=collection_name)
    else:
        print(f"Creating new collection: {collection_name}")

        schema = CollectionSchema([
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=3072),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="filename", dtype=DataType.VARCHAR, max_length=255),
        ], description="Internal_SOPs_AND_Regulations")

        collection = Collection(name=collection_name, schema=schema)

        collection.create_index(
            field_name="embedding",
            index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 128}}
        )

    required_partitions = ["Global_Regulatory"]
    existing_partitions = [p.name for p in collection.partitions]

    for partition in required_partitions:
        if partition not in existing_partitions:
            print(f"Creating missing partition: {partition}")
            collection.create_partition(partition)

    collection.load()
    print(f"Number of entities in the collection: {collection.num_entities}")
    return collection

def insert_into_milvus_lite(text_chunks, category, filename, partition_name="GlobalRegulations"):
    collection = create_or_get_milvus_lite_collection()

    entries = [create_document_entry(chunk, category, filename) for chunk in text_chunks]

    texts = [entry["text"] for entry in entries]
    categories = [entry["category"] for entry in entries]
    filenames = [entry["filename"] for entry in entries]
    embeddings = [entry["embedding"] for entry in entries]

    for i, embedding in enumerate(embeddings):
        if isinstance(embedding, str): 
            embeddings[i] = ast.literal_eval(embedding)
        embeddings[i] = [float(x) for x in embeddings[i]]  

    entities = [
        {"text": texts[i], "category": categories[i], "filename": filenames[i], "embedding": embeddings[i]}
        for i in range(len(texts))
    ]

    collection.insert(entities, partition_name=partition_name)
    collection.flush()
    print(f"Total number of records in collection: {collection.num_entities}")

def search_milvus_lite(query_text, top_k=10, user_id=None):
    print("Search milvus started......")
   
    query_embedding = get_hybrid_embedding(query_text)

    if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
        query_embedding = [item for sublist in query_embedding for item in sublist]
    elif isinstance(query_embedding, list):
        query_embedding = [float(x) if not isinstance(x, float) else x for x in query_embedding]
    else:
        print("Error: query_embedding should be a list.")

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    cfr_results = milvus_client.search(
        collection_name="Global_Regulatory",
        partition_names=["CFR"],
        data=[query_embedding],    
        limit=10,
        output_fields=["text", "category", "filename"]
    
    ) or [] 
    
    ich_results = milvus_client.search(
        collection_name="Global_Regulatory",
        partition_names=["ICH"],
        data=[query_embedding],
        limit=10,   
        output_fields=["text", "category", "filename"]
    ) or []
    
    ema_results = milvus_client.search(
        collection_name="Global_Regulatory",
        partition_names=["EMA"],
        data=[query_embedding],
        limit=10,   
        output_fields=["text", "category", "filename"]
    ) or []
    
    user_lib_results = []
    res_docs = []
    
    if user_id:
        ref_name = f"ref_{user_id}"
        res_docs = milvus_client.search(
            collection_name="Ref_Documents",
            partition_names=[ref_name],
            data=[query_embedding],
            limit=10,   
            output_fields=["text", "category", "filename"]
        ) or []

    def format_results(results):
        if not results or len(results) == 0:
            return []
        formatted_results = []
        for hit in results[0]:
            entity = hit.get('entity', {})
            text_value = entity.get("text", "No text found")
            if not isinstance(text_value, str):
                text_value = json.dumps(text_value)

            formatted_results.append({
                "score": hit.get('distance', 0.0),
                "text": text_value,
                "category": entity.get("category", "Unknown"),
                "filename": entity.get("filename", "Unknown")
            })
        return formatted_results

    cfr_formatted = format_results(cfr_results)
    ich_formatted = format_results(ich_results)
    ema_formatted = format_results(ema_results)
    internal_formatted = format_results(user_lib_results)
    res_docs = format_results(res_docs)
    combined_results = cfr_formatted + ich_formatted + internal_formatted + res_docs + ema_formatted
    # combined_results = ich_results + res_docs 
    return combined_results

def rag_generate(user_query, system_prompt, temperature=0.6, max_tokens=7000):
    print("@@@@#################")
    retrieved_chunks = search_milvus_lite(user_query, 10)
    if not retrieved_chunks:
        return {"message": "No relevant documents found."}

    combined_text = "\n".join([chunk["text"] for chunk in retrieved_chunks])
    full_prompt = f"Use the following context to answer the question.\n\n{combined_text}\n\nQuestion: {user_query}"
    print("&&&&&&&&&&&&&&&&",full_prompt)
    return deepseek_model.generate(
        prompt=full_prompt,
        system_prompt=system_prompt,
        temperature=temperature,
        max_new_tokens=max_tokens
    )

 
