from nltk.corpus import wordnet as wn
import nltk

import requests 

import google.generativeai as genai

from backEnd.ai.rag import get_relevant_context_from_db, generate_prompt

from NSP import run_model, reasume_base_data

nltk.download('wordnet')
GEMINI_API_KEY = "AIzaSyCFhiseeAtrOTwRc3Ib0-h4vqFha7KLgPM"
MODE_Model=1
m = None
device = 'cpu'


if MODE_Model == 1:
    max_iters = 50000
    n_layer = 32
    
    baseName = "model" # "RLHF"
    savingBasePath = f"save/model_nn_{str(n_layer)}_gen_{str(max_iters)}"
    model_name = f'{baseName}_nn_{str(n_layer)}_gen_{str(max_iters)}.pt'
    csv_file_name = f'{baseName}_loss_nn_{str(n_layer)}_gen_{str(max_iters)}.csv'    
    
    # Model
    # m = reasume_base_data(f"./{savingBasePath}/model_{model_name}", "cpu")
    m = reasume_base_data(f"./{savingBasePath}/RLHF_{model_name}", "cpu")






def agent_routine(query, keyWords, mode=MODE_Model):
    print(f"mode: {mode}")
    # WordNet
    keyWords = [wn.synsets(word)[0].name().split(".")[0] if wn.synsets(word) != [] else '' for word, qnt in keyWords.items()]
    keyWords.remove('')
    contextList = ", ".join(keyWords) 

    # RAG
    docs = get_relevant_context_from_db(query)

    # Build response
    question = generate_prompt(query, docs, contextList)
    print(f"\n\nQuestion: {question}\n\n")

    # Question to Model
    response = "No response"
    if mode == 0: response = get_gemini_response(question)
    else: response = get_model_response(question)

    # Register data to DB
    """TODO"""

    # Return
    return response


def get_model_response(prompt):
    if m == None: return "No model loaded"
    else: return run_model(m , device, str(prompt), decoded=True)


def get_gemini_response(prompt):
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(model_name='gemini-pro')
    answere = model.generate_content(prompt)
    return answere.text






if __name__ == "__main__":
    query = "How work blockchain ?"
    kw = {"blockchain": 1, "work": 2}

    resp = agent_routine(query, kw, 1)
    print(f"\nresp: {resp}\n")