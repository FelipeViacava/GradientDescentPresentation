from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

parser = reqparse.RequestParser()
parser.add_argument('data')


app = Flask(__name__)
model = pickle.load('model.pickle')
@app.route('/Predict')
def predict(self): # rewrite the post method inherited by Resource to use our own model
    args = parser.parse_args()
    X = np.array(args['data']) # deserializes the predictors from a string (how it was stored with pickle in a json), into a list and into a NumPy array using the parser
    prediction = model.predict(X) # predicts just as scikitlearn usually does
    return jsonify(prediction.tolist()) # returns serialize array of predicted labels back to the application that requested the api

if __name__ == '__main__':
    app.run(debug=True)