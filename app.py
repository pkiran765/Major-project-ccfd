from flask import Flask,request, url_for, redirect, render_template
import json
import numpy as np
from load_model import new_model
import pdb

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template("ccfd.html")


@app.route('/predict',methods=['POST','GET'])
def predict():
   
    data = request.form.values()
    print (data)
    data = list(data)
    print("after list conversion")
    print(data)
    
    data = list(map(float,data)) 
    print(type(data[0]))
    data = np.asarray(data).astype(float)
    data = list(map(float,data))
    data = np.array([data])
    ans = new_model.predict(data)
    print(ans[0][0])
    ans=ans[0][0]>0.5
    result = "Fraud" if ans else "Normal"
    
    return render_template('result.html', classValue = "class value - '{0}'".format(str(result)))

if __name__ == '__main__':
    app.run(debug=True)