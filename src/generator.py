import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2:7b"

def call_llm(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)

    if response.status_code != 200:
        return "LLM调用失败"
    
    data = response.json()

    return data["response"]

def parse_llm_output(text):

    answer = ""
    evidence = []

    lines = text.split("\n")

    for line in lines:

        if line.startswith("abswer"):
            answer = line.replace("answer:", "").strip()

        if line.startswith("-"):
            evidence.append(line.replace("-", "").strip())

    if answer == "":
        answer = text.strip()

    return {
        "status": "success",
        "answer": answer,
        "evidence": evidence
    }

def generate_answer(question, prompt, contexts):

    llm_output = call_llm(prompt)

    result = parse_llm_output(llm_output)

    return result
