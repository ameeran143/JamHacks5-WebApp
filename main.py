import streamlit as st
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier

st.set_page_config(layout="wide")

# Create a title and a sub title
st.write("""
# Incoming UW Student Guide
Short description of the website
""")

# Reading in the data
data = pd.read_csv("UW Data - Sheet1.csv")
print(data.head)

# CREATING THE MODEL
# Some of the values on the dataset are not integers, this converts them
#le = preprocessing.LabelEncoder()
le = HashingVectorizer()

# Coverting:
Status = le.fit_transform(list(data["Status"]))
Program = le.fit_transform(list(data["Program"]))
Grade = le.fit_transform(list(data["Grade"]))
ApplicantType = le.fit_transform(list(data["Applicant Type"]))

# Defining what the model is predicting for
predict = "Accepted"

x = list(zip(Program, Grade, ApplicantType))  # Features: things that are being used to find out if the person is accepted
y = list(Status)  # Label: What we are training the model to predict


x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# Training the Model
model = KNeighborsClassifier(n_neighbors=8)
model.fit(x_train, y_train)

# Printint out the accuracy
accuracy = model.score(x_test, y_test)
st.write(accuracy)


# This model is approximatley 90% accurate based off the given data we have at the moment.

# Getting User Input
def get_user_input():
    st.write("Prediction Model")
    Program = st.selectbox(
        'What program are you applying to?',
        ('Accounting and Financial Management', 'Architectural Engineering', 'Biomedical Engineering',
         'Chemical Engineering', 'Civil Engineering', 'Computer Engineering', 'Computer Science',
         'Computer Science and Financial Management', 'Computer Science/BBA', 'Electrical Engineering',
         'Environmental Engineering', 'Global Business and Digital Arts', 'Kinesiology', 'Life Science',
         'Management Engineering', 'Mathematics', 'Mathematics/BBA', 'Mechanical Engineering',
         'Mechatronics Engineering', 'Physical Sciences', 'Software Engineering', 'Systems Design Engineering'))
    Grade = st.slider("Grade (Top 6 Average)", 0.00, 100.00, 50.00)
    ApplicantType = st.selectbox("What type of Applicant are you?", ('101', '105'))

    le = preprocessing.LabelEncoder()

    user_data = {
        'Program': le.fit_transform(Program),
        'Grade': le.fit_transform(Grade),
        'ApplicantType': le.fit_transform(ApplicantType)
    }

    # Transofmring data into features
    features = pd.DataFrame(user_data, index=[0])
    return features


# Create a variable to store user input
user_input = get_user_input()

# Setting a subheader and then displaying user input
st.subheader("User input:")
st.write(user_input)

predicton = model.predict(user_input)
st.write(predicton)



