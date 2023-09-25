# -*- coding: utf-8 -*-
"""
Created on Fri Feb  24 20:11:32 2023

@author: Saketh
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(
    open('C:/Users/Saketh/Desktop/Multi_Disease_Prediction/Heart/heart_disease_model.sav', 'rb'))

# Creating a function for prediction


def heart_disease_prediction(input_data):

    # Changing input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)


def main():

    # Title
    st.title("Heart Disease Prediction")

    # Getting input data from user
    age = st.number_input('Age', step=1, max_value=110)
    sex = st.selectbox('Sex (female=0, male=1)', (0, 1))
    cp = st.selectbox('Chest pain type', (0, 1, 2, 3))
    trestbps = st.number_input('Resting Blood Pressure', step=1)
    chol = st.number_input('Serum Cholestoral in mg/dl', step=1)
    fbs = st.selectbox('Fasting blood sugar', (0, 1))
    restecg = st.number_input('Resting Electrocardiographic results', step=1)
    thalach = st.number_input('Maximum Heart Rate achieved', step=1)
    exang = st.number_input('Exercise Induced Angina', step=1)
    oldpeak = st.number_input('ST depression induced by exercise', step=0.1)
    slope = st.number_input('Slope of the peak exercise ST segment')
    ca = st.selectbox('Major vessels colored by flourosopy', (0, 1, 2, 3))
    thal = st.selectbox(
        'thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', (0, 1, 2))

    # page title
    st.title('Heart Disease Prediction using ML')

    # Code for Prediction
    heart_diagnosis = ''

    # Creating a button
    if st.button('Heart Disease Test Result'):
        heart_prediction = loaded_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if (heart_prediction[0] == 0):
            heart_diagnosis = 'This person does not have a Heart Disease'
        else:
            heart_diagnosis = 'This person has Heart Disease'
    st.success(heart_diagnosis)


if __name__ == '__main__':
    main()
