# Ollama-Running-DeepSeek-R1-Locally
 

# RAG-Based Document Q&A System

A Retrieval Augmented Generation (RAG) system built with Streamlit that enables intelligent document querying and answering.

## 🚀 Features

### Document Processing
- Multi-PDF file upload support
- Text extraction using PyMuPDFLoader
- Smart text chunking (400 characters with 100 character overlap)

### Vector Storage
- Implements Chroma DB for vector storage
- Ollama-based text embeddings
- Persistent storage with metadata and unique IDs
- Data stored at './demo-reg-chroma'

### Query Processing
The system processes queries through three main steps:
1. Vector database search for relevant content
2. Similar document retrieval
3. Answer generation using deepseek-r1 language model

### User Interface
Built with Streamlit, featuring:
- PDF upload sidebar
- Question input field
- Answer display area

## 🔄 System Flow

## 🤖 AI System Prompt
The AI assistant is configured to:
- Utilize only provided context
- Deliver structured responses
- Balance comprehensiveness with conciseness
- Acknowledge information gaps when present

## 🔍 Advanced Features
- CrossEncoder re-ranking capability (currently commented out)
- Enhanced result relevance potential

## 💡 Overview
This system functions as an intelligent document assistant, capable of understanding and answering questions based solely on the content of uploaded documents. It combines modern RAG architecture with an intuitive interface for efficient document interaction.

## 🛠️ Technical Requirements
- Python 3.10+
- Streamlit
- PyMuPDF
- Chroma DB
- Ollama
- deepseek-r1 model

## 📝 Note
The system includes a re-ranking feature using CrossEncoder that can be implemented for improved result relevance.


## Example 
![image](https://github.com/user-attachments/assets/30cfff9a-ba83-42fa-a905-a8b655797486)
![image](https://github.com/user-attachments/assets/a1c12673-29e7-45ad-881b-466e109cc015)


