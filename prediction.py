import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


def load_pickles(model_pickle_path, label_encoder_pickle_path):
    model_pickle_opener = open(model_pickle_path, "rb")
    model = pickle.load(model_pickle_opener)

    label_encoder_pickle_opener = open(label_encoder_pickle_path, "rb")
    label_encoder_dict = pickle.load(label_encoder_pickle_opener)

    return model, label_encoder_dict


def pre_process_data(df, label_encoder_dict):
    # df.drop("customerID", axis=1, inplace=True)
    for col in df.columns:
        if col in list(label_encoder_dict.keys()):
            column_le = label_encoder_dict[col]
            df.loc[:, col] = column_le.transform(df.loc[:, col])
        else:
            continue
    return df


def make_predictions(processed_df, model):
    prediction = model.predict(processed_df)
    return prediction


def generate_predictions(test_df):
    model_pickle_path = "./churn_prediction_model.pkl"
    label_encoder_pickle_path = "./churn_prediction_label_encoders.pkl"

    model, label_encoder_dict = load_pickles(
        model_pickle_path, label_encoder_pickle_path
    )

    processed_df = pre_process_data(test_df, label_encoder_dict)
    prediction = make_predictions(processed_df, model)
    return prediction


if __name__ == "__main__":
    # make the application
    st.title("Customer Churn Prediction")
    st.text("Enter customer data.")

    # making customer data input
    gender = st.selectbox("Select customer's gender :", ["Female", "Male"])
    senior_citizen_input = st.selectbox(
        "Is customer a senior citizen? :", ["No", "Yes"]
    )
    if senior_citizen_input == "Yes":
        senior_citizen = 1
    else:
        senior_citizen = 0
    partner = st.selectbox("Does the customer have a partner? :", ["No", "Yes"])
    dependents = st.selectbox("Does the customer have dependents? :", ["Yes", "No"])
    tenure = st.slider(
        "How many months has the customer been with the company? :",
        min_value=0,
        max_value=72,
        value=24,
    )
    phone_service = st.selectbox(
        "Does the customer have phone service? :", ["No", "Yes"]
    )
    multiple_lines = st.selectbox(
        "Does the customer have multiple lines? :", ["No", "Yes", "No phone service"]
    )
    internet_service = st.selectbox(
        "What type of internet service does the customer have? :",
        ["No", "DSL", "Fiber optic"],
    )
    online_security = st.selectbox(
        "Does the customer have online security? :",
        ["No", "Yes", "No internet service"],
    )
    online_backup = st.selectbox(
        "Does the customer have online backup? :", ["No", "Yes", "No internet service"]
    )
    device_protection = st.selectbox(
        "Does the customer have device protection? :",
        ["No", "Yes", "No internet service"],
    )
    tech_support = st.selectbox(
        "Does the customer have tech support? :", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.selectbox(
        "Does the customer have streaming TV? :", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Does the customer have streaming movies? :",
        ["No", "Yes", "No internet service"],
    )
    contract = st.selectbox(
        "What kind of contract does the customer have? :",
        ["Month-to-month", "Two year", "One year"],
    )
    paperless_billing = st.selectbox(
        "Does the customer have paperless billing? :", ["No", "Yes"]
    )
    payment_method = st.selectbox(
        "What is the customer's payment method? :",
        [
            "Mailed check",
            "Credit card (automatic)",
            "Bank transfer (automatic)",
            "Electronic check",
        ],
    )
    monthly_charges = st.slider(
        "What is the customer's monthly charge? :", min_value=0, max_value=118, value=50
    )
    total_charges = st.slider(
        "What is the total charge of the customer? :",
        min_value=0,
        max_value=8600,
        value=2000,
    )
    input_dict = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }
    input_data = pd.DataFrame([input_dict])

    if st.button("Predict Churn"):
        pred = generate_predictions(input_data)
        if bool(pred):
            st.error("Customer will churn!")
        else:
            st.success("Customer not predicted to churn")
