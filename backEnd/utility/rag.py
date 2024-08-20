import signal
import re
import sys
import requests

from nltk.corpus import wordnet as wn
import nltk

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from model.nsp import get_model_response
from model.llama import get_answere_from_llama

nltk.download('wordnet')
API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
API_key = "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"
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

def generate_rag_prompt(query, context):
    escaped = context.replace("'", "").replace('"', "").replace("\n", "")
    escaped = (re.sub(r'\[\d\]', "", escaped)).replace("[]", "")
    escaped = (re.sub(r'\[\d\d\]', "", escaped)).replace("[]", "")
    escaped = (re.sub(r'\[\d\d\d\]', "", escaped)).replace("[]", "")
    prompt = ("""
                You're a helpful and informative chatbot that answers questions using the information provided in the reference context. 
                Make sure your response is a complete sentence that covers all relevant details. 
                Remember, you're talking to someone who might not be familiar with technical terms, so try to explain things in a simple and friendly way. 
                If the information isn't related to the question, you can ignore it.
                    Question: {query}
                    Context: {context}
              
                    Answer: 
              """).format(query=query, context=context)
    return prompt






def rutine_response(query):
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    
    return get_model_response(prompt)

def ai_response(query):
    context = get_relevant_context_from_db(query)
    prompt = generate_rag_prompt(query=query, context=context)
    return get_answere_from_llama(prompt)



if __name__=="__main__":
    print("--------------------------------------------------------------------------------------------------------------")
    print(get_answere_from_llama("Can you quickly introduce yourself"))
    print("--------------------------------------------------------------------------------------------------------------")

    while True:
        print("--------------------------------------------------------------------------------------------------------------")
        print("What would you like to ask:")
        query = input(">>> ")
        print(f".\n{ai_response(query)}\n")


