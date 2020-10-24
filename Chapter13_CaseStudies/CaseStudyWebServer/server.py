import os
import pickle

from flask import Flask, request, jsonify
import numpy as np


PATH = os.path.abspath(os.path.join(__file__, os.pardir))
MODEL_PATH = os.path.join(PATH, "model.pkl")

APP = Flask(__name__)
MODEL = pickle.load(open(MODEL_PATH, "rb"))


@APP.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    pred = MODEL.predict([np.array(data['x'])])
    output = pred[0]
    return jsonify(output)


if __name__ == '__main__':
    APP.run(port=5000, debug=True)
