import json
from llamaapi import LlamaAPI

# Initialize the SDK
llama = LlamaAPI("<your_api_token>")


def get_answere_from_llama(query):
    # Build the API request
    api_request_json = {
        "messages": [
            {"role": "user", "content": query},
        ],
        "stream": False,
    }

    # Execute the Request
    return llama.run(api_request_json)
    
