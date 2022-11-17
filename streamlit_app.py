import streamlit as st
import pandas as pd

st.write("Here we create data using a table:")
st.write(
    pd.DataFrame({"first column": [1, 2, 3, 4], "second column": [10, 20, 30, 40]})
)


data = pd.read_csv("./data/training_data.csv")

st.write("Churn Data")
st.write(data)

st.write("How many customers in the dataset churned?")
target_bins = data.loc[:, "Churn"].value_counts()
st.bar_chart(target_bins)
