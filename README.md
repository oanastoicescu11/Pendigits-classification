# XGBoost model for classification on the Pen-Based Recognition of Handwritten Digits Data Set. 
Author: E. Alpaydin, Fevzi. Alimoglu
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Pen-Based+Recognition+of+Handwritten+Digits) - 1998-07-01

Xgboost implements **gradient boosted decision tree**. Details:https://xgboost.readthedocs.io/en/latest/.
For more details on the model training and evaluation, please refer to the IPYNB file.

### Instructions on how to install and run the api (tested on Windows 10 Home):

prerequisite: docker installed

in Powershell:

1. navigate to the working directory `cd '<your path>\pendigits\scripts'`
    
2. build docker image `docker build -t pendigits .`
    
3. run docker container `docker run -d -p 5000:5000 pendigits`
    
4. open in browser http://localhost:5000/

### Querying the api

1. to get predictions on the test data: http://localhost:5000/predict


2. to predict on new data entry: http://localhost:5000/predict_new/?input1=100&input10=7&input11=80&input12=0&input13=80&input14=24&input15=26&input16=26&input2=100&input3=46&input4=89&input5=13&input6=64&input7=0&input8=34&input9=25

    Replace the values after "=" with new values in the url query. If there are missing values or values in the wrong format, the app imputs 0.
    
    An example of query can be found in the file 'Query string example.txt'
    
    Important! The app assumes the columns are in the following order: **'input1', 'input10', 'input11', 'input12', 'input13', 'input14',
        'input15', 'input16', 'input2', 'input3', 'input4', 'input5', 'input6', 'input7', 'input8', 'input9'.** 
    

3. To imput new data as json: the app can be tested with Postman: https://www.postman.com/product/rest-client/.
   Send a request to http://localhost:5000/predict_new, copy the text from 'json example.txt' into the postman request, as seen in the screenshot below.


![Postman_screenshot.png](Postman_screenshot.png)
