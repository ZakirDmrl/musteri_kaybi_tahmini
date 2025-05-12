import streamlit as st
import pickle
import numpy as np

# Modeli yükle
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("SVC RBF + SelectKBest Tahmin Uygulaması")
st.write("En iyi 8 özellik ile sınıflandırma yapar.")

# Girişleri al
credit_score = st.number_input("Credit Score", value=650)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", value=35)
tenure = st.number_input("Tenure", value=5)
balance = st.number_input("Balance", value=50000.0)
num_of_products = st.number_input("Number of Products", value=1)
has_cr_card = st.selectbox("Has Credit Card?", [0, 1])
is_active = st.selectbox("Is Active Member?", [0, 1])
salary = st.number_input("Estimated Salary", value=60000.0)

# Encode input
geo_france = 1 if geography == "France" else 0
geo_germany = 1 if geography == "Germany" else 0
gen = 1 if gender == "Male" else 0

input_array = np.array([[
    credit_score,
    geo_france,
    geo_germany,
    gen,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active,
    salary
]])

# Tahmin butonu
if st.button("Tahmin Et"):
    # Model 8 özelliğe göre eğitildiyse sadece o kısımları seç
    # Örneğin aşağıdaki gibi slice kullanabilirsin:
    input_selected = input_array[:, :8]  # Sadece ilk 8 özelliği aldı (k=8 için)

    prediction = model.predict(input_selected)[0]
    probability = model.predict_proba(input_selected)[0]

    st.write(f"**Tahmin (Exited):** {prediction}")
    st.write(f"**Olasılıklar [Kalmadı, Çıktı]:** {probability}")
