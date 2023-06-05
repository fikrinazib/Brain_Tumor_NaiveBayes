import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import streamlit as st


#data.csv belom ada
data = pd.read_csv('Brain Tumor.csv')

# preProcessing 
data = data.drop(data[['Image']],axis=1)

# Memisahkan label dan fitur 
X = data.drop(columns='Class', axis=1)
Y = data['Class']

# standarisasi data
scaler = StandardScaler()
scaler.fit(X)
standarized_data = scaler.transform(X)
X = standarized_data
Y = data['Class']

# Pisah data training dan test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=8)

# Naive Bayes
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Form

st.title("Brain Tumor Detection")
col1, col2, col3 = st.columns(3)
with col1:
    Mean = st.text_input("Input Nilai Mean",0)
with col2:
    Variance = st.text_input("Input Nilai Variance",0)
with col3:
    Standard_Deviation = st.text_input("Input Nilai Standard Deviation",0)
with col1:
    Entropy = st.text_input("Input Nilai Entropy",0)
with col2:
    Skewness = st.text_input("Input Nilai Skewness",0)
with col3:
    Kurtosis = st.text_input("Input Nilai Kurtosis",0)
with col1:
    Contrast = st.text_input("Input Nilai Contrast",0)
with col2:
    Energy = st.text_input("Input Nilai Energy",0)
with col3:
    ASM = st.text_input("Input Nilai ASM",0)
with col1:
    Homogeneity = st.text_input("Input Nilai Homogeneity",0)
with col2:
    Dissimilarity = st.text_input("Input Nilai Dissimilarity",0)
with col3:
    Correlation = st.text_input("Input Nilai Correlation",0)
with col1:
    Coarseness = st.text_input("Input Nilai Coarseness",0)
submit = st.button('Submit')

completeData = np.array([Mean, Variance, Standard_Deviation, Entropy, Skewness, Kurtosis, Contrast, Energy, ASM, Homogeneity, Dissimilarity, Correlation, Coarseness ]).reshape(1, -1)
scaledData = scaler.transform(completeData)

if submit: 
    prediction = classifier.predict(scaledData)
    if prediction == 0:
        st.success('Pasien tidak terkena tumor')
    else :
        st.error('Pasien terkena tumor')
