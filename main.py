from flask import Flask
from flask import request
from flask import render_template

from Agent import agent_routine


app = Flask(__name__)


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/agent", methods=["POST"])
def AgentCall():
    print("In !!!")
    jsn = request.get_json()
    print(f"json: {jsn["query"]}")
    print(f"json: {len(jsn["keyWords"])}")

    if jsn["query"] != "" and len(jsn['keyWords']) > 0:
        print(jsn["query"])
        print(jsn['keyWords'][0])

        response = agent_routine(jsn["query"], jsn['keyWords'][0], mode=0)

        return {"status": "true", "response": response}, 201

    return {"status": "fail"}, 411


@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html'), 404




if __name__ == '__main__':  
   app.run()