import streamlit as st
import pandas as pd
import joblib


model=joblib.load("xgboost_model.joblib")
preprosses=joblib.load("preprocessor.joblib")






credit_score= st.number_input('Credit Score', min_value=300, max_value=850, value=600, step=10)
geography= st.selectbox('Geography', options=['France', 'Spain', 'Germany'], index=0)
gender= st.selectbox("Gender", options=["Male","Female"], index=0)
age= st.number_input('Age', min_value=18, max_value=100, value=30, step=1)
tenure= st.number_input('Tenure', min_value=0, max_value=10, value=5, step=1)
balance= st.number_input('Balance', min_value=0.0, max_value=250000.0, value=10000.0, step=100.0)
num_of_products= st.number_input('Number of Products', min_value=1, max_value=4, value=1, step=1)
has_cr_card= st.selectbox('Has Credit Card', options=[0, 1], index=1)
is_active_member= st.selectbox('Is Active Member', options=[0, 1],  index=1)
estimated_salary= st.number_input('Estimated Salary', min_value=0.0, max_value =200000.0, value=50000.0, step=1000.0)
status= st.button('Predict')
if status:
    ################################################################
    input_data = [credit_score, geography,gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]#
 

    column_names = [
        'CreditScore',
        'Geography',
        'Gender',
        'Age',
        'Tenure',
        'Balance',
        'NumOfProducts',
        'HasCrCard',
        'IsActiveMember',
        'EstimatedSalary'
    ]

    df = pd.DataFrame([input_data], columns=column_names)
    ###############################################################
    transformed=preprosses.transform(df)
    predict=model.predict(transformed)
    predict_propa=model.predict_proba(transformed)
    result = "shurn"if predict ==1 else "stayed"
    result_proba=1 if predict ==1 else 0
    ###############################################################
    st.write(f"the client will {result}")
    st.write(f"the probability of client will {result} is {predict_propa[0][result_proba]*100:.2f} %")
