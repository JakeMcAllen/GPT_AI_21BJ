import os
import re
import signal 
import sys 

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceHubEmbeddings




GEMEINI_API_KEY = "AIzaSyCFhiseeAtrOTwRc3Ib0-h4vqFha7KLgPM"
HUGGINFACE_TOKEN = "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"

extraInputInfo =  """You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information. \n
      Try to responde to this question: "{user_input}" \n
      The main topic of question are: {contextWords} \n
      This is the recommended document thath you can use to generate response: {relevant_document} \n
      Compile a recommendation to the user based on the recommended activity and the user input."""

os.environ['HUGGINGFACEHUB_API_TOKEN'] =  HUGGINFACE_TOKEN
os.environ['HF_TOKEN']  =  HUGGINFACE_TOKEN



def signal_handler(sign, frame):
    print('\nBot stopped. ')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


def generate_prompt(query, context, keyWords):
    escaped = context.replace("'", "").replace('"', "").replace("\n", " ") #.replace("[%d]","").replace("[%d%d]","").replace("[%d%d%d]","")
    [escaped := re.sub(fr'\[{r'\d'*i}\]', '', escaped) for i in range(1, 5)]
    return extraInputInfo.format(relevant_document=escaped, user_input=query, contextWords=keyWords)



def get_relevant_context_from_db(query):
    context = ""
    embedding_function = HuggingFaceHubEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="./chroma_db_nccn", embedding_function=embedding_function)
    search_results = vector_db.similarity_search(query, k=6)
    for result in search_results:
        context += result.page_content + "\n"
    return context


def RAG_sys(query, keyWords):
    prompt = get_relevant_context_from_db(query)
    return generate_prompt(query, prompt, keyWords)


if __name__ == "__main__":
    print("What would you like to ask: ")

    while True:
        query = input(">>> ")
        prompt = generate_prompt( get_relevant_context_from_db(query) )

        print(f"content: {prompt}")
