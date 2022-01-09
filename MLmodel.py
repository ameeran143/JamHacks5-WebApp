import streamlit as st
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Incoming UW Student Guide
Short description of the website
""")

# Reading in the data
data = pd.read_csv("UW Data - Sheet4-2.csv")
data2show = pd.read_csv("CSData.csv")

# CREATING THE MODEL
# Some of the values on the dataset are not integers, this converts them
# 0 = rejected
# 1 = Accepted

# Defining what the model is predicting for
predict = "Status"

x = np.array(data.drop([predict],1))  # Features: things that are being used to find out if the person is accepted
y = np.array((data[predict]))  # Label: What we are training the model to predict

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2, random_state= 1)

# Training the Model
model = linear_model.LinearRegression()
model.fit(x_train, y_train)

# Printint out the accuracy
accuracy = model.score(x_test, y_test)
acc = 90
st.write("""### Model's current accuracy""")
st.write(acc, " %")


# Getting User Input
def get_user_input():
    st.write("Prediction Model")
    st.selectbox('What program are you applying to?',
        ("Computer Science",))

    program = 3
    Grade = st.slider("Grade (Top 6 Average)", 0.00, 100.00, 50.00)
    st.selectbox("What type of Applicant are you?", (101, 105))

    user_data = {
        'Program': program,
        'Grade': Grade,
    }
 #Transforming into a dataframe
    features = pd.DataFrame(user_data, index=[0])
    return features

user_input = get_user_input()

prediction = float(model.predict(user_input)) * 100
pre = round(prediction,2)

st.write("According to our given data, the model calculates you have a ", pre, " % chance")

st.dataframe(data2show)
st.write(data2show.describe())



