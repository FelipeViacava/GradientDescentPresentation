from flask import Flask, jsonify
from flask_restful import Api, Resource, reqparse
import pickle
import numpy as np
import json

app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()
parser.add_argument('data')


class RF_Classifier(Resource): # creates modified Resource class
    def post(self): # rewrite the post method inherited by Resource to use our own model
        args = parser.parse_args()
        X = np.array(json.loads(args['data'])) # deserializes the predictors from a string (how it was stored with pickle in a json), into a list and into a NumPy array using the parser
        prediction = model.predict(X) # predicts just as scikitlearn usually does
        return jsonify(prediction.tolist()) # returns serialize array of predicted labels back to the application that requested the api

api.add_resource(RF_Classifier, '/RandomForest')

if __name__ == '__main__': 
    # Load model 
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    app.run(debug=True)