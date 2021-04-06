from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))


@app.route('/')
def hello_world():

    return render_template("diabetes.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    
    int_features=[x for x in request.form.values()]
    user_details = int_features[0:3]
    temp = int_features[3:]
    model_features = [float(x) for x in temp]
    final=[np.array(model_features)]
    print(int_features)
    print(final)
    prediction=model.predict_proba(final)
    output='{0:.{1}f}'.format(prediction[0][1], 2)

    if output == str(0):
        return render_template('result.html',pred=' Name : {}\n, Age : {}\n, Gender : {} \n high chance of developing diabetes'.format(user_details[0],user_details[1], user_details[2]))
    else:
        return render_template('result.html',pred=' Name : {}\n, Age : {}\n, Gender : {} \n Low chance of developing diabetes'.format(user_details[0],user_details[1], user_details[2]))


if __name__ == '__main__':
    app.run(debug=True)