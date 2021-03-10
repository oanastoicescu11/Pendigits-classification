# this api provides an interface for xgb predictions. Route /predict offers predictions on a test set of 54 rows.
# route /predict_new takes data in json format as input, or parses the url query to make predictions on new data.
# The app assumes the values in the url query are in the order of the columns.
# The order of columns is (['input1', 'input10', 'input11', 'input12', 'input13', 'input14',
#        'input15', 'input16', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9']

# Dependencies
import sys
from flask import Flask, request, jsonify
import pickle
import traceback
import pandas as pd
import numpy as np

# API definition
app = Flask(__name__)


def preprocess_data(data, columns):
    # get the same colnames as in the model
    data = data.reindex(columns=columns, fill_value=0)
    data = data.astype(int)
    return data

# functions to load pickle files
def load_model():
    with open('model.pkl', 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    print('Model loaded')
    return model


def load_columns():
    with open('columns.pkl', 'rb') as pickle_file:
        columns = pickle.load(pickle_file)
    print('Columns loaded')
    return columns


def load_test_data():
    with open('mydata.pkl', 'rb') as pickle_file:
        mydata = pickle.load(pickle_file)
    print('Test data loaded')
    return mydata


def load_response():
    with open('response.pkl', 'rb') as pickle_file:
        response = pickle.load(pickle_file)
    print('Test classes loaded')
    return response


@app.route('/', methods=['GET'])
def index():
    return 'Use /predict to get predictions on test data. Use /predict_new to input data'


@app.route('/predict', methods=['GET'])
# here we get the predictions made on the test data (90 samples loaded from the pickle file mydata)
def predict():
    # load the pickles
    try:
        model = load_model()
    except:
        return 'model.pkl file is missing'

    try:
        columns = load_columns()
    except:
        return 'columns.pkl file is missing'

    try:
        mydata = load_test_data()
    except:
        return 'mydata.pkl file is missing'

    try:
        response = load_response()
    except:
        return 'response.pkl file is missing'

    try:
        # preprocess the data
        data = preprocess_data(pd.DataFrame(mydata), columns)
        print(data.shape)
        # make predictions
        prediction = list(model.predict(data))
        response = list(response)
        return jsonify({'prediction': str(prediction), 'real': str(response)})

    except:

        return jsonify({'trace': traceback.format_exc()})



@app.route('/predict_new/', methods=['GET', 'POST'])
# here we can input json or parse the url query
def predict_new():
    # load necessary pickle files
    try:
        model = load_model()
    except:
        return 'model.pkl file is missing'

    try:
        columns = load_columns()
    except:
        return 'columns.pkl file is missing'

    # request json
    json_ = request.json
    print(json_)
    if json_:
        # preprocess data
        data = preprocess_data(pd.DataFrame(json_), columns)
        print(data.shape)
        # make predictions
        prediction = list(model.predict(data))

        return jsonify({'prediction': str(prediction)})
    else:
        # parse url args if json not present. The app assumes arguments are in order of columns!
        args = request.args
        print(args)
        if args:
            # preprocess data
            data = preprocess_data(pd.DataFrame(data=[args]), columns)
            # make predictions
            prediction = list(model.predict(data))

            return jsonify({'prediction': str(prediction)})
        else:
            return 'Type a url query string'


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])  # This is for a command-line input
    except:
        port = 12345  # If you don't provide any port the port will be set to 12345

    app.run(port=port, debug=True)
