# -*- coding: utf-8 -*-
"""
Created on Sun May 22 11:53:51 2022

@author: Saketh
"""

import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn import datasets
import requests
from streamlit_lottie import st_lottie

# loading the saved models

diabetes_model = pickle.load(open(
    'C:/Users/Saketh/Desktop/Multi_Disease_Prediction/Diabetes/diabetes_model.sav', 'rb'))

heart_disease_model = pickle.load(open(
    'C:/Users/Saketh/Desktop/Multi_Disease_Prediction/Heart/V1/heart_disease_model.sav', 'rb'))

parkinsons_model = pickle.load(open(
    'C:/Users/Saketh/Desktop/Multi_Disease_Prediction/Parkinsons/parkinsons_model.sav', 'rb'))


# sidebar for navigation
with st.sidebar:

    selected = st.selectbox('Multiple Disease Prediction System',
                            ['Home',
                             'Diabetes Prediction',
                             'Heart Disease Prediction',
                             'Parkinsons Prediction'],)


if (selected == 'Home'):
    st.title("Multiple Disease Prediction using Machine Learning")

    def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_url_hello = "https://assets3.lottiefiles.com/packages/lf20_0ssane8p.json"
    lottie_hello = load_lottieurl(lottie_url_hello)
    st_lottie(lottie_hello, key="hello")

# Diabetes Prediction Page
if (selected == 'Diabetes Prediction'):
    # page title
    st.title('Diabetes Prediction using ML')
    # getting the input data from the user
    colm1, colm2, colm3 = st.columns(3)

    with colm1:
        Pregnancies = st.text_input('Number of Pregnancies')

    with colm2:
        Glucose = st.text_input('Glucose Level')

    with colm3:
        BloodPressure = st.text_input('Blood Pressure value')

    with colm1:
        SkinThickness = st.text_input('Skin Thickness value')

    with colm2:
        Insulin = st.text_input('Insulin Level')

    with colm3:
        BMI = st.text_input('BMI value')

    with colm1:
        DiabetesPedigreeFunction = st.text_input(
            'Diabetes Pedigree Function value')

    with colm2:
        Age = st.text_input('Age of the Person')

    # code for Prediction
    diab_diagnosis = ''

    # creating a button for Prediction

    if st.button('Diabetes Test Result'):
        diab_prediction = diabetes_model.predict(
            [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])

        if (diab_prediction[0] == 1):
            diab_diagnosis = 'The person is diabetic'
        else:
            diab_diagnosis = 'The person is not diabetic'

    st.success(diab_diagnosis)


# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):

    # page title
    st.title('Heart Disease Prediction using ML')

    colm1, colm2, colm3 = st.columns(3)

    with colm1:
        age = st.number_input('Age', step=1, max_value=110)

    with colm2:
        sex = st.selectbox('Sex (female=0, male=1)', (0, 1))

    with colm3:
        cp = st.selectbox('Chest pain type', (0, 1, 2, 3))

    with colm1:
        trestbps = st.number_input('Resting Blood Pressure', step=1)

    with colm2:
        chol = st.number_input('Serum Cholestoral in mg/dl', step=1)

    with colm3:
        fbs = st.selectbox('Fasting blood sugar', (0, 1))

    with colm1:
        restecg = st.number_input(
            'Resting Electrocardiographic results', step=1)

    with colm2:
        thalach = st.number_input('Maximum Heart Rate achieved', step=1)

    with colm3:
        exang = st.number_input('Exercise Induced Angina', step=1)

    with colm1:
        oldpeak = st.number_input(
            'ST depression induced by exercise', step=0.1)

    with colm2:
        slope = st.number_input('Slope of the peak exercise ST segment')

    with colm3:
        ca = st.selectbox('Major vessels colored by flourosopy', (0, 1, 2, 3))

    with colm1:
        thal = st.selectbox(
            'thal: 0 = normal; 1 = fixed defect; 2 = reversable defect', (0, 1, 2))

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        if (heart_prediction[0] == 1):
            heart_diagnosis = 'The person is having heart disease'
        else:
            heart_diagnosis = 'The person does not have any heart disease'
        st.success(heart_diagnosis)


# Parkinson's Prediction Page
if (selected == "Parkinsons Prediction"):

    # page title
    st.title("Parkinson's Disease Prediction using ML")

    colm1, colm2, colm3, col4, col5 = st.columns(5)

    with colm1:
        fo = st.text_input('MDVP:Fo(Hz)')

    with colm2:
        fhi = st.text_input('MDVP:Fhi(Hz)')

    with colm3:
        flo = st.text_input('MDVP:Flo(Hz)')

    with col4:
        Jitter_percent = st.text_input('MDVP:Jitter(%)')

    with col5:
        Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')

    with colm1:
        RAP = st.text_input('MDVP:RAP')

    with colm2:
        PPQ = st.text_input('MDVP:PPQ')

    with colm3:
        DDP = st.text_input('Jitter:DDP')

    with col4:
        Shimmer = st.text_input('MDVP:Shimmer')

    with col5:
        Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')

    with colm1:
        APQ3 = st.text_input('Shimmer:APQ3')

    with colm2:
        APQ5 = st.text_input('Shimmer:APQ5')

    with colm3:
        APQ = st.text_input('MDVP:APQ')

    with col4:
        DDA = st.text_input('Shimmer:DDA')

    with col5:
        NHR = st.text_input('NHR')

    with colm1:
        HNR = st.text_input('HNR')

    with colm2:
        RPDE = st.text_input('RPDE')

    with colm3:
        DFA = st.text_input('DFA')

    with col4:
        spread1 = st.text_input('spread1')

    with col5:
        spread2 = st.text_input('spread2')

    with colm1:
        D2 = st.text_input('D2')

    with colm2:
        PPE = st.text_input('PPE')

    # code for Prediction
    parkinsons_diagnosis = ''

    # creating a button for Prediction
    if st.button("Parkinson's Test Result"):
        parkinsons_prediction = parkinsons_model.predict(
            [[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5, APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

        if (parkinsons_prediction[0] == 1):
            parkinsons_diagnosis = "The person has Parkinson's disease"
        else:
            parkinsons_diagnosis = "The person does not have Parkinson's disease"

    st.success(parkinsons_diagnosis)
