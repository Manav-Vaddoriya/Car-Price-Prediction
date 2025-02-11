import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open('car_predict.pkl', 'rb') as file:
    model = pickle.load(file)

# loading the dataset
car = pd.read_csv('Cleaned_Car_data.csv')

st.title("Car Price Prediction App 🚗💰")

companies = sorted(car['company'].unique())
model_names = sorted(car['name'].unique())
year = sorted(car['year'].unique())
fuel_type = sorted(car['fuel_type'].unique())

inp_comp=st.selectbox('Enter the company:',['']+companies)

models_for_inp_comp = car[car['company']==inp_comp]['name']
inp_model=st.selectbox('Enter the model name:',['']+models_for_inp_comp)
inp_year=st.selectbox('Enter the year:',['']+year)
inp_fuel_type=st.selectbox('Enter the fuel type:',['']+fuel_type)
inp_kms_driven=st.text_input('Enter the kms driven:')

button = st.button("Predict the price")
if button:
    prediction = model.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                                            data=np.array([inp_model,inp_comp,inp_year,inp_kms_driven,inp_fuel_type]).reshape(1,5)))
    st.markdown(f"<h3 style='text-align: center; color: blue;'>Prediction Result: {round(prediction[0])}</h3>", unsafe_allow_html=True)
