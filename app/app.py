import base64
import io
from PIL import Image
from flask import Flask, jsonify, make_response, request

# Map labels and categories
labels_dict = {}

app = Flask(__name__)

@app.route('/image/predict/', methods=['POST'])
def predict():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload["image"][0]
    image_binary = base64.b64decode(image_b64)
    # Convert the base64 image to PIL Image object
    image_buf = io.BytesIO(image_binary)
    image = Image.open(image_buf)
    # Preprocess the image
    # Predict with the model
    # Format and return the response
    response = {"category": "Arboria", "probability": 0.5}
    return make_response(jsonify(response))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)