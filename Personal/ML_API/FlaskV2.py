from asyncore import read
from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

parser = reqparse.RequestParser()
parser.add_argument('data')

app = Flask(__name__)
model = pickle.load(open('model.pickle','wb'))

@app.route('/Predict')
def predict(self): 
    args = parser.parse_args()
    X = np.array(args['data'])
    prediction = model.predict(X)
    return jsonify(prediction.tolist())

if __name__ == '__main__':
    app.run(debug=True)