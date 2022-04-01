import base64
import requests

filename = "../images/arborio.jpg"
api_url = "http://192.168.0.23:5000/serving/predict"
with open(filename, "rb") as img:
    img_b64 = base64.b64encode(img.read())
    
payload = {"image": img_b64}
response = requests.post(url=api_url, data=payload)
print(response.status_code)  
print(response.text)  
