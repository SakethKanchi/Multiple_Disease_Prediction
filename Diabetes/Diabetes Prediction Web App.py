# -*- coding: utf-8 -*-
"""
Created on Fri Jan  20 21:08:41 2023

@author: Saketh
"""

import numpy as np
import pickle
import streamlit as st

# Loading the saved model
loaded_model = pickle.load(
    open('C:/Users/Saketh/Desktop/Multi_Disease_Prediction/Diabetes/trained_model.sav', 'rb'))

# Creating a function for prediction


def diabetes_prediction(input_data):

    # Changing input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array for prediction
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def main():

    # Title
    st.title("Diabetes Prediction")

    # Getting input data from user
    Pregnancies = st.text_input("Number of Pregnancies")
    Glucose = st.text_input("Glucose Level value")
    BloodPressure = st.text_input("Diastolic blood pressure (mm Hg)")
    SkinThickness = st.text_input("Triceps skin fold thickness (mm)")
    Insulin = st.text_input("2-Hour serum insulin value (mu U/ml)")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input(
        "Diabetes Pedigree Function value")
    Age = st.text_input("Age of the Person(years)")

    # Code for Prediction
    diagnosis = ''

    # Creating a button
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction(
            [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)


if __name__ == '__main__':
    main()
