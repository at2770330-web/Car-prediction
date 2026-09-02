from email import encoders

from click import File
from idna import encode
from matplotlib import category
import streamlit as st
import pickle
import numpy as np
import pandas as pd

from streamlit_car_ML import Fuel_type, Manufacturer
st.title(" 🚗CAR PRICE PREDICTION 🚗")
st.write("Accurate Price Prediction In All Car Models")
#load saved model  and encoders
@st.cache_resource
def load_saved_model():
    with open('car_model.pkl','rb') as file:
        saved_data = pickle.load(file)
    st.markdown("---")
    return saved_data

try:
    saved_data=load_saved_model()
    ml_model=saved_data['model']
    encoders=saved_data['encoders']
    
    #from input:
    col1,col2=st.columns(2)
    with col1:
        manufacturer=st.selectbox("manufacturer",encoders['Manufacturer'].classes_)
        model=st.selectbox("model",encoders['Model'].classes_)
        category=st.selectbox("category",encoders['Category'].classes_)
        prod_year=st.number_input("prod_year",min_value=1990,max_value=2026,value=2015)
        mileage=st.number_input("mileage (in km)",min_value=0,value=50000)
    with col2:
        fuel_type=st.selectbox("fuel_type",encoders['Fuel_type'].classes_)
        leather_interior=st.selectbox("leather_interior",encoders['Leather_interior'].classes_)
        color=st.selectbox("color",encoders['Color'].classes_)
        levy=st.number_input("levy",min_value=0,value=0)
        engine_volume=st.number_input("engine_volume",min_value=0.0,value=2.0,step=0.1)
        cylinders=st.number_input("cylinders",min_value=1,max_value=12,value=4)
        gear_box_type=st.number_input("gear_box_type",min_value=0,max_value=5,value=1)
        drive_wheels=st.number_input("drive_wheels",min_value=0,max_value=5,value=1)
        doors=st.number_input("doors",min_value=1,max_value=6,value=4)
        wheel=st.number_input("wheel",min_value=0,max_value=10,value=1)
        airbags=st.number_input("airbags",min_value=0,max_value=16,value=4)
    #from predict botton
    if st.button("Predict Price",use_container_width=True):
        #input data Encode
        input_dict={
            'ID':0,
            'Prod_year':prod_year,
            'Levy':levy,
            'Mileage':mileage,
            'Engine_volume':engine_volume,
            'Cylinders':cylinders,
            'Doors':doors,
            'Airbags':airbags,
            'Manufacturer':encoders['Manufacturer'].transform([manufacturer])[0],
            'Fuel_type':encoders['Fuel_type'].transform([fuel_type])[0],
            'Leather_interior':encoders['Leather_interior'].transform([leather_interior])[0],
            'Color':encoders['Color'].transform([color])[0],
            'Model':encoders['Model'].transform([model])[0],
            'Category':encoders['Category'].transform([category])[0],
            'Gear_box_type':gear_box_type,
            'Drive_wheels':drive_wheels,
            'Wheel':wheel
        }
        input_df=pd.DataFrame([input_dict])
        # Use the exact feature schema stored by the fitted model.
        input_df = input_df[ml_model.feature_names_in_]
        #model prediction
        prediction=ml_model.predict(input_df)[0]
        #display result
        if prediction<0:
            st.warning("predict result negative pls input the negative parameters")
        else:
            st.success(f"Estimate price:{prediction:,.2f}")
except FileNotFoundError:
    st.error('car_model.pkl')
    

                
            
        
                