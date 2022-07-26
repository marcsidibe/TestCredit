# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

st.write('# Lapplication qui prédit l accord de crédit')


#Collecte le profil d'entrée
st.sidebar.header(" Les caractéristiques du client ")

def client_caract_client():
    Gender=st.sidebar.selectbox('Gender',('Male', 'Female'))
    Married=st.sidebar.selectbox('Married',('Yes', 'No'))
    Dependents = st.sidebar.selectbox('Nombre d enfants',('0','1','2','3+'))
    Education=st.sidebar.selectbox('Diplômé',('Graduate','Not Graduate'))
    Self_Employed = st.sidebar.selectbox('Salarié',('Yes', 'No'))
    ApplicantIncome = st.sidebar.slider('Salaire du client',150,4000,200)
    CoapplicantIncome=st.sidebar.slider('Salaire du conjoint',0,40000,2000)
    LoanAmount=st.sidebar.slider('Montant du crédit',9.0,700.0,200.0)
    Loan_Amount_Term=st.sidebar.selectbox('Durée du crédit',(360.0, 120.0,240.0,180.0,60.0, 300.0, 84.0,12.0))
    Credit_History = st.sidebar.selectbox('Historique du crédit', (1.0, 0.0))
    Property_Area = st.sidebar.selectbox('Property_Area', ('Urban', 'Rural', 'Semiurban'))


    data = {
        'Gender':Gender,
        'Married':Married,
        'Dependents':Dependents,
        'Education':Education,
        'Self_Employed':Self_Employed,
        'Credit_History':Credit_History,
        'Property_Area':Property_Area,
        'ApplicantIncome':ApplicantIncome,
        'CoapplicantIncome':CoapplicantIncome,
        'LoanAmount':LoanAmount,
         'Loan_Amount_Term':Loan_Amount_Term
    }
    profil_client = pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_caract_client()

#Transformation des donnees d entree en donnees adaptees
# Importation des donnees
df=pd.read_csv('train.csv')
credit_input=df.drop(columns=['Loan_ID','Loan_Status'])
donnee_entree=pd.concat([input_df,credit_input],axis=0)

var_num=['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
var_cat=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']

df_cat=donnee_entree[var_cat]
df_num=donnee_entree[var_num]
df_cat=pd.get_dummies(df_cat, drop_first=True)

df_encoded=pd.concat([df_cat, df_num], axis= 1)
#for col in var_cat:
 #   dummy=pd.get_dummies(donnee_entree[col],drop_first=True)
  #  donnee_entree=pd.concat([dummy,donnee_entree], axis=1)
   # del donnee_entree[col]

st.write(df_encoded)

#Prendre uniquement la premiere ligne
df_encoded=df_encoded[:1]


#Afficher les donnees transformees
st.subheader('Les caracteristiques transformees')
st.write(df_encoded)

#Importattion du modèle
load_model= pickle.load(open('prevision_credit2.pkl', 'rb'))


# Appliquer le modele sur le profil du client
prevision=load_model.predict(df_encoded)

st.subheader('Resultat de la prevision')
if(prevision):
    st.write("**On n'accorde pas le crédit à ce client**")
else:
    st.write("**On accorde le crédit à ce client**")