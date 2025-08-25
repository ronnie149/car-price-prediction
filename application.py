import streamlit as st

import pickle
import pandas as pd
import numpy as np

app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year,fuel_types=fuel_type)


st.title("🚗 Car Price Prediction App")

companies = sorted(car['company'].unique())
car_models = sorted(car['name'].unique())
years = sorted(car['year'].unique(), reverse=True)
fuel_types = car['fuel_type'].unique()

company = st.selectbox("Select Company", companies)
car_model = st.selectbox("Select Car Model", car_models)
year = st.selectbox("Select Year", years)
fuel_type = st.selectbox("Select Fuel Type", fuel_types)
driven = st.number_input("Kilometers Driven", min_value=0)

if st.button("Predict Price"):
    input_df = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5))
    prediction = model.predict(input_df)
    st.success(f"Estimated Price: ₹{np.round(prediction[0], 2):,.2f}")
