# -*- coding: utf-8 -*-
"""
Created on Fri Feb  24 20:11:32 2023

@author: Saketh
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import StandardScaler


def user_input_features():

    # Title
    st.title("Heart Disease Prediction")

    global age
    # Getting input data from user
    age = st.number_input('Enter your age: ', step=1)
    sex = st.selectbox('Sex', (0, 1))
    cp = st.selectbox('Chest pain type', (0, 1, 2, 3))
    tres = st.number_input('Resting blood pressure: ', step=1)
    chol = st.number_input('Serum cholestoral in mg/dl: ', step=1)
    fbs = st.selectbox('Fasting blood sugar', (0, 1))
    res = st.number_input('Resting electrocardiographic results: ', step=1)
    tha = st.number_input('Maximum heart rate achieved: ', step=1)
    exa = st.selectbox('Exercise induced angina: ', (0, 1))
    old = st.number_input('oldpeak ', step=0.1)
    slope = st.number_input(
        'The slope of the peak exercise ST segmen: ')
    ca = st.selectbox('Number of major vessels', (0, 1, 2, 3))
    thal = st.selectbox('thal', (0, 1, 2))

    data = {'age': age,
            'sex': sex,
            'cp': cp,
            'trestbps': tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach': tha,
            'exang': exa,
            'oldpeak': old,
            'slope': slope,
            'ca': ca,
            'thal': thal
            }
    features = pd.DataFrame(data, index=[0])
    return features


input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

df = pd.concat([input_df, heart_dataset], axis=0)

# Encoding of ordinal features
df = pd.get_dummies(
    df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1]  # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('./Random_forest_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)
while age > 0:
    if prediction[0] == 1:
        st.success("You have a heart disease")
        break
    else:
        st.success("You don't have a heart disease")
        break

st.subheader('Prediction Probability')
st.write(prediction_proba)
