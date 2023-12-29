from flask import Flask,render_template,request,redirect
import pickle
import pandas as pd
import numpy as np
 
app=Flask(__name__)
model=pickle.load(open('wine_quality_predictor.pkl','rb'))
car=pd.read_csv('winequality-white.csv')

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    data9 = request.form['i']
    arr = np.array([[data1, data2, data3, data4,data5, data6, data7, data8,data9]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)
if __name__=='__main__':
    app.debug=True
    app.run()
