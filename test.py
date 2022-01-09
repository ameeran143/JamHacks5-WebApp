
import streamlit as st
import pandas as pd
import sklearn
import numpy as np
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier


data = pd.read_csv("UW Data - Sheet1.csv")
data2 = data.iloc[5:10]
print(data2.head)

st.write("""test""")

Program = st.selectbox(
        'What program are you applying to?',
        ('Accounting and Financial Management', 'Architectural Engineering', 'Biomedical Engineering', 'Chemical Engineering', 'Civil Engineering', 'Computer Engineering', 'Computer Science', 'Computer Science and Financial Management', 'Computer Science/BBA', 'Electrical Engineering', 'Environmental Engineering', 'Global Business and Digital Arts', 'Kinesiology', 'Life Science', 'Management Engineering', 'Mathematics', 'Mathematics/BBA', 'Mechanical Engineering', 'Mechatronics Engineering', 'Physical Sciences', 'Software Engineering', 'Systems Design Engineering' ))