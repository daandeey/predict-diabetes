import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn import preprocessing
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest.sav', 'rb'))
scaler = pickle.load(open('standard_scaler.sav', 'rb'))
SKEWED_COLUMNS = ['Pregnancies', 'SkinThickness', 'Insulin', 'DiabetesPedigreeFunction', 'Age']
cols=['Pregnancies',
      'Glucose',
      'BloodPressure',
      'SkinThickness',
      'Insulin',
      'BMI',
      'DiabetesPedigreeFunction',
      'Age']

def preprocess(df, scaler):
    # impute data with 0 value with median from training set
    df['Glucose'] = df['Glucose'].replace(0, 117.0)
    df['BloodPressure'] = df['BloodPressure'].replace(0, 72.0)
    df['SkinThickness'] = df['SkinThickness'].replace(0, 23.0)
    df['Insulin'] = df['Insulin'].replace(0, 30.5)
    df['BMI'] = df['BMI'].replace(0, 32.0)
    
    df[SKEWED_COLUMNS] = np.log(df[SKEWED_COLUMNS]+1)
    df = scaler.transform(df)
    
    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    feature_list = request.form.to_dict()
    
    df_feature_list = pd.DataFrame(data=np.array(list(feature_list.values())).reshape(1,-1), columns=feature_list.keys())
    df_feature_list = df_feature_list.astype(float)
    final_features = preprocess(df_feature_list, scaler)
    prediction = model.predict(final_features)
    output = int(prediction[0])
    if output == 1:
        text = "Diabetes"
    else:
        text = "Healthy"

    return render_template('index.html', prediction_text='Patient is {}'.format(text))


if __name__ == "__main__":
    app.run(debug=True)