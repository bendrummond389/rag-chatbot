from flask import Flask, request, jsonify
from langchain_community.llms import Ollama

app = Flask(__name__)

cached_llm = Ollama(model='llama3')

# response = llm.invoke('why is the sky blue')

# print(response)

@app.route("/ai", methods=["POST"])
def ai_post():
    print("Post /ai called")
    json_content = request.json

    if json_content is None:
        return jsonify({"error": "Invalid JSON"}), 400

    query = json_content.get("query")

    response = cached_llm.invoke(query)

    response_answer = {
        "answer": response
    }
    return response

@app.route("/pdf", methods=["POST"])
def pdf_post():
    file = request.files["file"]
    file_name = file.filename

    if file_name is None:
        return jsonify({"error": "File name not found"}), 400

    save_file = "pdf/" + file_name
    file.save(save_file)

    response = {
        "status": "Success",
        "filename": file_name
    }

    return response






    # if query:
    #         # Invoke the LLM
    #         response = llm.invoke(query)
    #         print("response: ", response)
    #         return jsonify({"response": response})
    # else:
    #     return jsonify({"error": "No query provided"}), 400





def start_app():
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    start_app()
