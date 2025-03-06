import streamlit as st
import pickle
import numpy as np

# Streamlit UI
st.title("🪴I own an Iris flowers Shop🌱")
st.subheader("Enter the parameters for Iris flower.")
sepal_length = st.number_input("Sepal Length(cm)", min_value=4.3, max_value= 7.9, value= 5.84)
sepal_width = st.number_input("Sepal Width(cm)", min_value = 2.0, max_value = 4.4, value = 3.057)
petal_length = st.number_input("Petal Length(cm)", min_value = 1.0, max_value = 6.9)
petal_width = st.number_input("Petal Width(cm)", min_value = 0.1, max_value = 2.5)

with open('knn_iris_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
    
with open('knn_iris_model_scaled.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

if st.button("Predict"):
    input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_features = loaded_scaler.transform(input_features)
    value = loaded_model.predict(input_features)
    if value == 0:
        st.write("It is Setosa!")
    elif value == 1:
        st.write("It is Versicolor.")
    else:
        st.write("It is virginica.")
