from nltk.corpus import wordnet as wn
import nltk

import requests 


nltk.download('wordnet')

extraInputInfo =  """You are a bot that makes recommendations for activities. You answer in very short sentences and do not include extra information.
      This is the recommended activity:
      {relevant_document}
      The user input is:
      {user_input}
      Use the contect words:  
      {contextWords}
      Compile a recommendation to the user based on the recommended activity and the user input."""

API_URL = "https://api-inference.huggingface.co/models/openai-community/gpt2"
API_key = "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"



def Agent(question, context):
        # WordNet
        context = [wn.synsets(word)[0].name().split(".")[0] if wn.synsets(word) != [] else '' for word, qnt in context.items()]
        context.remove('')
        contextList = ", ".join(context) 

        # RAG
        """TODO"""
        docs


        # Build response
        question = extraInputInfo.format(relevant_document=docs ,user_input=question, contextWords=contextList)


        # Question to Model
        response = get_llama_RLHF_response(question)
        # response = get_model_response(question)

        # Register data to DB
        """TODO"""

        # Return
        return response


def get_model_response(msg: str):
    pass



def get_llama_RLHF_response(msg: str, min_length=256, max_length=500, print_response=False, eliminate_input=True) -> str:
    payload = { "inputs": msg, "top_k": max_length, "min_length": min_length, max_length:"max_length", "temperature": 0.8, "max_time": 40, "do_sample": True }
    headers = {"Authorization": f"Bearer {API_key}"} 
    errorCount = 0
    
    while True:
        try:
            response = requests.post(API_URL, headers=headers, json=payload)
            
            if response.status_code != 200: raise Exception(f"response:{response} and status: {response.json()}")
            if print_response: print(response.json(), end="\n\n")
            return response.json()[0]["generated_text"][len(msg) + 1:] if eliminate_input else response.json()[0]["generated_text"]
            
        except Exception as e:
            print(f"Error: {e}" if errorCount == 0 else "." if errorCount%100000 == 0 else "", end="")
            errorCount += 1