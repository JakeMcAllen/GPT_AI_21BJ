import os
import signal
import sys
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


GEMEINI_API_KEY = "AIzaSyCFhiseeAtrOTwRc3Ib0-h4vqFha7KLgPM"
NUMBER_OF_RESULT_RAG = 6


def signal_handler(sig, frame):
    print("\nThank for usign Alita")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=NUMBER_OF_RESULT_RAG)
    for result in search_results:
        context += result.page_content + "\n"
    return context



while True:
    print("--------------------------------------------------------------------------------------------------------------")
    print("What would you like to ask")
    query = input(">>> ")
    context = get_relevant_context_from_db(query)
    print(f".\n{context}\n")
