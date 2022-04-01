from flask import Flask, jsonify, make_response,request
from api_utils import BasmatinetPrediction


predictor = BasmatinetPrediction()
         
app = Flask(__name__)


@app.route('/serving/predict', methods=['POST'])
def prediction_pipeline():
    # Get the image in base 64 and decode it
    payload = request.form.to_dict(flat=False)
    image_b64 = payload["image"][0]
    # Pass it through the inference pipeline
    response = predictor.inference_pipeline(image_b64)
    return make_response(jsonify(response))

@app.route('/serving/healthcheck', methods=['GET'])
def healthcheck():
    #return make_response(jsonify({"status": "ok"}))
    return 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)