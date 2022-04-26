import base64
import requests
import argparse

front_parser = argparse.ArgumentParser(
    description='For an input image return the prediction of a dockerized application.')
front_parser.add_argument('--filename', type=str, default='../images/arborio.jpg',
                          help='Path of the file of which we want the prediction')
front_parser.add_argument('--host-ip', type=str, default='0.0.0.0',
                          help='IP address of the host of the prediction api')

args = front_parser.parse_args()
filename = args.filename
api_url = 'http://' + args.host_ip + ':5001/serving/predict'

with open(filename, 'rb') as img:
    img_b64 = base64.b64encode(img.read())

payload = {'image': img_b64}
response = requests.post(url=api_url, data=payload)

print('Prediction status code: ', response.status_code)
print('Prediction: ', response.text)
