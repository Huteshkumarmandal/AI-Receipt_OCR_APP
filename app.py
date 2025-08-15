import os
from flask import Flask, request, render_template, jsonify
import pandas as pd
import json, re
import ollama
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create folders if not exist
os.makedirs("uploads", exist_ok=True)

app = Flask(__name__)

# Helper function to extract JSON from Ollama + LangChain response
def extract_json_from_text(text):
    def get_response(response):
        template = """Extract the following receipt details from the provided text response and return them as a structured JSON object.

        Input text:
        {response}

        Return only JSON.
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOllama(model="llama3", temperature=0)
        chain = prompt | llm | StrOutputParser()
        return chain.invoke({"response": response})

    result = get_response(text)
    json_match = re.search(r"```\n(.*?)\n```", result, re.DOTALL)
    if json_match:
        return json.loads(json_match.group(1))
    else:
        return {}

@app.route("/", methods=["GET", "POST"])
def index():
    json_data = None
    df_rows = []
    df_columns = []

    if request.method == "POST":
        file = request.files["image"]
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)

        # Step 1: OCR using Ollama
        response = ollama.chat(
            model="llama3.2-vision",
            messages=[{"role": "user", "content": "get all the data from the image", "images": [file_path]}],
        )
        cleaned_text = response['message']['content'].strip()

        # Step 2: Extract structured JSON
        parsed_data = extract_json_from_text(cleaned_text)
        json_data = json.dumps(parsed_data, indent=2)

        # Step 3: Convert items to DataFrame
        if parsed_data.get("Items"):
            df = pd.DataFrame(parsed_data["Items"])
            # Add general receipt details
            details = {
                "Company": parsed_data.get("Company"),
                "Bill To": f"{parsed_data['Bill To']['Name']}, {parsed_data['Bill To']['Address']}" if parsed_data.get('Bill To') else "",
                "Date": parsed_data.get("Date"),
                "Subtotal": parsed_data.get("Subtotal"),
                "Tax": parsed_data.get("Sales Tax"),
                "Total": parsed_data.get("Total"),
                "Payment Method": parsed_data.get("Payment Instructions", {}).get("Method")
            }
            for key, value in details.items():
                df[key] = value

            df_rows = df.values.tolist()
            df_columns = df.columns.tolist()

    return render_template("index.html", json_data=json_data, df_rows=df_rows, df_columns=df_columns)

if __name__ == "__main__":
    app.run(debug=True)
