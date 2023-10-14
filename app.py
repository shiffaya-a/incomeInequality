import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

def encode(input_val, feats):
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value

def getPredict_Model(data,model):
    return model.predict(data)

st.set_page_config(page_title="Income Inequality Prediction App", layout="centered")

#creating option list for dropdown menu
options_gender = ['Female', 'Male']
options_education=['High school graduate', '12th grade no diploma', 'Children',
       'Bachelors degree(BA AB BS)', '7th and 8th grade', '11th grade',
       '9th grade', 'Masters degree(MA MS MEng MEd MSW MBA)',
       '10th grade', 'Associates degree-academic program',
       '1st 2nd 3rd or 4th grade', 'Some college but no degree',
       'Less than 1st grade', 'Associates degree-occup /vocational',
       'Prof school degree (MD DDS DVM LLB JD)', '5th or 6th grade',
       'Doctorate degree(PhD EdD)']
options_marital_status=['Widowed', 'Never married', 'Married-civilian spouse present',
       'Divorced', 'Married-spouse absent', 'Separated',
       'Married-A F spouse present']
options_race=['White', 'Black', 'Asian or Pacific Islander',
       'Amer Indian Aleut or Eskimo', 'Other']
options_is_hispanic=['All other', 'Mexican-American', 'Central or South American',
       'Mexican (Mexicano)', 'Puerto Rican', 'Other Spanish', 'NA',
       'Cuban', 'Do not know', 'Chicano']
options_employment_commitment=['Not in labor force', 'Children or Armed Forces',
       'Full-time schedules', 'PT for econ reasons usually PT',
       'Unemployed full-time', 'PT for non-econ reasons usually FT',
       'PT for econ reasons usually FT', 'Unemployed part- time']
options_industry_code_main=['Not in universe or children', 'Hospital services', 'Retail trade',
       'Finance insurance and real estate',
       'Manufacturing-nondurable goods', 'Transportation',
       'Business and repair services', 'Medical except hospital',
       'Education', 'Construction', 'Manufacturing-durable goods',
       'Public administration', 'Agriculture',
       'Other professional services', 'Mining',
       'Utilities and sanitary services', 'Private household services',
       'Personal services except private HH', 'Wholesale trade',
       'Communications', 'Entertainment', 'Social services',
       'Forestry and fisheries', 'Armed Forces']
options_household_summary=['Householder', 'Child 18 or older', 'Child under 18 never married',
       'Spouse of householder', 'Nonrelative of householder',
       'Other relative of householder',
       'Group Quarters- Secondary individual',
       'Child under 18 ever married']
options_tax_status=['Head of household', 'Single', 'Nonfiler', 'Joint both 65+',
       'Joint both under 65', 'Joint one under 65 & one 65+']
options_household_summary=['Householder', 'Child 18 or older', 'Child under 18 never married',
       'Spouse of householder', 'Nonrelative of householder',
       'Other relative of householder',
       'Group Quarters- Secondary individual',
       'Child under 18 ever married']
options_citizenship=['Native', 'Foreign born- Not a citizen of U S',
       'Foreign born- U S citizen by naturalization',
       'Native- Born abroad of American Parent(s)',
       'Native- Born in Puerto Rico or U S Outlying']

options_household_stat=['Householder', 'Nonfamily householder', 'other',
       'Secondary individual']


features=['age', 'gender', 'education', 'marital_status', 'race', 'is_hispanic',
       'employment_commitment', 'wage_per_hour', 'working_week_per_year',
       'industry_code', 'industry_code_main', 'occupation_code',
       'total_employed', 'household_summary', 'tax_status', 'gains', 'losses',
       'stocks_status', 'citizenship', 'household_stat',
       'importance_of_record']


st.markdown("<h1 style='text-align: center;'>Income Inequality Prediction App</h1>", unsafe_allow_html=True)
def main():
    with st.form('prediction_form'):

        st.subheader("Enter the input for following features:")
        age=st.slider("Age",1,100,value=0,format="%d")
        gender=st.selectbox("Gender",options=options_gender)
        education =st.selectbox("Education",options=options_education) 
        marital_status =st.selectbox("Marital Status",options=options_marital_status) 
        race =st.selectbox("Race",options=options_race) 
        is_hispanic =st.selectbox("Is Hispanic",options=options_is_hispanic) 
        employment_commitment =st.selectbox("employment_commitment",options=options_employment_commitment) 
        wage_per_hour =st.slider("Wage Per Hour",0,10000,value=0,format="%d")
        working_week_per_year=st.slider("Working_week_per_year",0,52,value=0,format="%d")
        industry_code =st.slider("industry_code",0,52,value=0,format='%d') 
        industry_code_main=st.selectbox("industry_code_main",options=options_industry_code_main)
        occupation_code=st.slider('Occupation Code',0,46,value=0,format='%d')
        total_employed=st.slider('Total Employed',0,6,value=0,format='%d')
        household_summary=st.selectbox("Household Summary",options=options_household_summary)
        tax_status=st.selectbox("Tax Status",options=options_tax_status)
        gains=st.slider('Gains',0,50000,value=0,format='%d')
        losses=st.slider('Losses',0,10000,value=0,format='%d')
        stocks_status=st.slider('stocks_status',0,100000,value=0,format='%d')
        citizenship=st.selectbox("Citizenship",options=options_citizenship)
        household_stat=st.selectbox("Household Status",options=options_household_stat)
        importance_of_record =st.slider('Importance_of_record',35,20000,value=0,format='%f')

        submit = st.form_submit_button("Predict")

        if submit:
            model=joblib.load(open(r'Model/Model.joblib',"rb"))

            gender = encode (gender,options_gender)  
            education = encode(education,options_education)       
            marital_status=encode(marital_status,options_marital_status)
            race=encode(race,options_race)
            is_hispanic =encode(is_hispanic,options_is_hispanic)
            employment_commitment=encode(employment_commitment,options_employment_commitment)
            industry_code_main=encode(industry_code_main,options_industry_code_main)
            household_summary=encode(household_summary,options_household_summary)
            tax_status=encode(tax_status,options_tax_status)
            citizenship=encode(citizenship,options_citizenship)
            household_stat=encode(household_stat,options_household_stat)
            
            
            data=np.array(features).reshape(1,-1)
            pred = getPredict_Model(data=data, model=model)
            if pred[0] == 0:
                result = 'Below Limit'
            else:
                result = 'Above Limit'
            st.write(f"The predicted severity is: {result}")

if __name__ == '__main__':
    main()