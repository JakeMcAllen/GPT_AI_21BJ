from flask import Flask, jsonify, request

from utility.rag import rutine_response, ai_response

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route('/getResponse', methods = ['POST'])
def login():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        json = request.get_json()
        # Call AI
        # response = rutine_response(json["query"])                      # AI made for TESI
        response = ai_response(json["query"])                          # External ai

        # Create json for output
        data = {'response': response}
        # Send output
        return jsonify(data), 201
    else:
        return jsonify({"error": 'Content-Type not supported!'}), 501


