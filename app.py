import pandas as pd 
import matplotlib.pyplot as plt 
import plotly.express as px 
import streamlit as st

@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv'
    df = pd.read_csv(url, sep=",")
    df_interim = df.copy()
    #Eliminating irrelevant data, not use
    drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
    df_interim.drop(drop_cols, axis = 1, inplace = True)

    #Dropping data with fare above 300
    df_interim.drop(df_interim[(df_interim['Fare'] > 300)].index, inplace=True)

    # Handling Missing Values in df_interim

    ## Fill missing AGE with Median of the survided and not survided is the same
    df_interim['Age'].fillna(df_interim['Age'].median(), inplace=True)

    ## Fill missing EMBARKED with Mode
    df_interim['Embarked'].fillna(df_interim['Embarked'].mode()[0], inplace=True)
    df = df_interim.copy()
    return df

df_ch = load_data()

st.title('Titanic dataset explorer')
st.subheader('Dataframe')
st.dataframe(df_ch)
st.subheader('Histograms')
col1, col2 = st.columns(2)
fig1 = px.histogram(df_ch, x='Survived')
fig2 = px.histogram(df_ch, x='Age')
col1.plotly_chart(fig1, use_container_width=True)
col2.plotly_chart(fig2, use_container_width=True)

# Heroku uses the last version of python, but it conflicts with 
# some dependencies. Low your version by adding a runtime.txt file
# https://stackoverflow.com/questions/71712258/
