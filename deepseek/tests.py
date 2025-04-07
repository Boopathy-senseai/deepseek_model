import requests
import json
 
url = "http://127.0.0.1:8000/generate/"
 
payload = json.dumps({
"prompt": "explain about deepseek and gpt models",
"system_prompt": "You are a helpful AI assistant",
"temperature": 0.7
})
headers = {
'Content-Type': 'application/json'
}
 
response = requests.request("POST", url, headers=headers, data=payload)
 
print(response.text)

