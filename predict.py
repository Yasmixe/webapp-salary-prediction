import streamlit as st 
import pickle 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()
regressor = data['model']
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")
    st.write("""### we need some information to predict the salary""")
    countries = (
   "United States of America",
   "Netherlands",
    "Italy",
    "Canada",
    "Germany",
    "Poland",
    "France",
    "Brazil",
    "Sweden",
    "Spain",
    "India",
    "Switzerland",
    "Australia",
    "Russian Federation",
     )
    education = (

    "Bachelor’s degree",
    "Master’s degree",
    "Less than a bachelors",
    "Post grad",
     )
    country = st.selectbox("Country", countries)
    educ = st.selectbox("Education Level", education)
    experience = st.slider("Years of Experience", 0, 50, 3)
    
    ok = st.button=("calculate Salary")
    if ok:
       x = np.array([[country, educ, experience ]])
       x[:, 0] = le_country.transform(x[:, 0])
       x[:, 1] = le_education.transform(x[:, 1])
       x= x.astype(float)
       salary = regressor.predict(x)
       st.subheader(f"the salary is ${salary[0]:.2f}")
