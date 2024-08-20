from flask import Flask, jsonify, request

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
        response = json["name"]
        
        # Create json for output
        data = {'response': response}
        # Send output
        return jsonify(data), 201
    else:
        return jsonify({"error": 'Content-Type not supported!'}), 501


