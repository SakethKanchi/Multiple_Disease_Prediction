import streamlit as st
import pandas as pd
from sklearn import datasets
import numpy as np
from streamlit_lottie import st_lottie
import requests

heart_data_csv = pd.read_csv("./heart.csv")
x = heart_data_csv[:0]
y = heart_data_csv.target
heart_data = heart_data_csv.drop(columns=['target'])
st.write("Shape of dataset", x)
# st.write(data)
st.write("Number of classes", len(np.unique(y)))


i = 0
expander = st.expander("Learn more", expanded=False)
if i == 0:
    expander.write("This is a very important dataset")
else:
    expander.write("Nyope")
# with expander:
#     st.info("Some info Here")
