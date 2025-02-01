import streamlit as st

import tempfile
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile

import chromadb
from chromadb.utils.embedding_functions.ollama_embedding_function import OllamaEmbeddingFunction
import ollama
from sentence_transformers import CrossEncoder
# rag = retrieval augmented generation
system_prompt = """
You are an AI assistant tasked with providing detailed answers based solely on the given context. Your goal is to analyze the information provided and formulate a comprehensive, well-structured response to the question.

context will be passed as "Context:"
user question will be passed as "Question:"

To answer the question:
1. Thoroughly analyze the context, identifying key information relevant to the question.
2. Organize your thoughts and plan your response to ensure a logical flow of information.
3. Formulate a detailed answer that directly addresses the question, using only the information provided in the context.
4. Ensure your answer is comprehensive, covering all relevant aspects found in the context.
5. If the context doesn't contain sufficient information to fully answer the question, state this clearly in your response.

Format your response as follows:
1. Use clear, concise language.
2. Organize your answer into paragraphs for readability.
3. Use bullet points or numbered lists where appropriate to break down complex information.
4. If relevant, include any headings or subheadings to structure your response.
5. Ensure proper grammar, punctuation, and spelling throughout your answer.

Important: Base your entire response solely on the information provided in the context. Do not include any external knowledge or assumptions not present in the given text.
"""
def process_document(uploaded_files: list[UploadedFile]):
    all_docs = []
    for uploaded_file in uploaded_files:
        temp_file = tempfile.NamedTemporaryFile('wb', suffix='.pdf', delete=False)
        temp_file.write(uploaded_file.read())
        temp_file.close()  # Close the file before loading

        loader = PyMuPDFLoader(temp_file.name)
        docs = loader.load()
        os.unlink(temp_file.name)
        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "?", "!", " ", ""]
    )
    return text_splitter.split_documents(all_docs)




def get_vector_collection():
    ollama_ef = OllamaEmbeddingFunction(
        url='http://localhost:11434/api/embeddings',
        model_name="nomic-embed-text:latest"  # Changed from models_name to model_name
    )

    chroma_client = chromadb.PersistentClient(
        path='./demo-reg-chroma'
    )
    
    return chroma_client.get_or_create_collection(
        name='rag_app',
        embedding_function=ollama_ef,
        metadata={'hnsw:space': 'cosine'}
    )


def add_to_vector_collection(all_splits:list[Document],file_name:str):
    collection = get_vector_collection()
    document,metadatas,ids = [],[],[]
    for idx,split in enumerate(all_splits):
        document.append(split.page_content)
        metadatas.append(split.metadata)  # Remove the curly braces
        ids.append(f"{file_name}_{idx}")

    collection.upsert(
        documents = document,
        metadatas = metadatas,
        ids = ids
    )
    st.success("Data added to the vector store!")

#process the query
def query_collection (prompt:str,n_results:int =10):
    collection = get_vector_collection()
    results = collection.query(
        query_texts=[prompt],
        n_results = n_results,
    )
    return results



def call_llm (context:str, prompt:str):
    response = ollama.chat(
        model="deepseek-r1:latest",
        stream=True,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": f"Context: {context}, Question: {prompt}",
            },
        ],
    )
    for chunk in response:
        if chunk["done"] is False:
            yield chunk["message"]["content"]
        else:
            break





# def re_ranking_encoders(documents:list[str]):
#     relevent_text = ""
#     relevant_text_ids =[]
#     encoder_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
#     ranks = encoder_model.rank(prompt, documents,top_k =3)
#     st.write(ranks)
#     for rank in ranks:
#         relevent_text += documents[rank['corpus_id']]
#         relevant_text_ids.append(rank['corpus_id'])

#     st.write(relevent_text)
#     st.divider()

#     return relevant_text_ids,relevent_text 






def get_document_loader(file_path: str, file_type: str):
    """Return appropriate loader based on file type"""
    loaders = {
        'pdf': PyMuPDFLoader,
        'txt': TextLoader,
        'docx': UnstructuredWordDocumentLoader,
        'csv': CSVLoader,
        'json': JSONLoader
    }
    
    loader_class = loaders.get(file_type.lower())
    if not loader_class:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return loader_class(file_path)





if __name__ == "__main__":
    with st.sidebar:
        st.set_page_config(page_title='RAG Question Answering')
        st.header('RAG Question Answering')

        upload_files = st.file_uploader(
            '** Upload PDF files for Ana **', 
            type=['pdf'],
            accept_multiple_files=True
        )
        process = st.button("Process..")

   
    for upload_file in upload_files:
        normalize_uploaded_file_name = upload_file.name.translate(
            str.maketrans({
                "-": "_",
                ".": "_",
                " ": "_"
            })
        )
        all_splits = process_document(upload_files)
        add_to_vector_collection(all_splits, normalize_uploaded_file_name)
    st.header("Ask your question")
    prompt  = st.text_input('Enter your question here')
    ask = st.button('Ask')

    if ask and prompt:
        results = query_collection(prompt)
        context = results.get('documents')[0]
        # relevant_text, relevant_text_ids = re_ranking_encoders(context)
        response = call_llm(context=context,prompt = prompt)    
        st.write(response)

    
