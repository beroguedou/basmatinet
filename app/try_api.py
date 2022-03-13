import base64
import requests

filename = "arborio"
api_url = "http://192.168.0.23:5000/image/predict/"
with open(filename+".jpg", "rb") as img:
    img_b64 = base64.b64encode(img.read())
    
payload = {"image": img_b64}
response = requests.post(url=api_url, data=payload)
print(response.status_code)  
print(response.text)  
#with open(filename+".txt", "w") as fic:
#    fic.write(string)