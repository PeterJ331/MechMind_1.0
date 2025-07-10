# deepseek_client.py

import requests

class DeepSeekClient:
    def __init__(self, model="deepseek-r1:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def ask(self, prompt):
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: {response.status_code} {response.text}"
