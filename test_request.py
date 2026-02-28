import base64
import requests

with open("test.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

url = "https://princymaheshwari--workshop-finder-pipeline-infer-dev.modal.run"

payload = {
    "image_b64": img_b64,
    "query": "wrench"
}

r = requests.post(url, json=payload)
print(r.json())