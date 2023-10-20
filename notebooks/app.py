import streamlit as st
import numpy as np
import pandas as pd
import pickle

df = pd.read_parquet('cleaned_data.parquet')
model_pipeline = pickle.load(open('grad_reg_pipeline.pkl','rb'))

st.title("House Price Prediction")

location = st.selectbox('location', df['location'].unique())
total_sqft = st.number_input('total sqft')
bath = st.number_input('bath')
bhk = st.number_input('bhk')

input_data = (location, total_sqft, bath, bhk)

if st.button('Predict Price'):
    # changing the data into a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    pred = model_pipeline.predict(input_data_reshaped)

    st.success(
        "The price of the house in the location given product {} and details is :-{}".format(
            location,pred
        )
    )
    st.title("The predicted price of this configuration is " + str(pred[0]))


