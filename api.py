from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

joblib_model = joblib.load('model/gbr_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    pred = joblib_model.predict(data)
    return jsonify({'prediction': pred.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
