from operator import index
import streamlit as st
import plotly.express as px
from pycaret import classification  
from pycaret import regression 
from pandas_profiling import ProfileReport
from PIL import Image 
import pandas as pd
from pandas_profiling import profile_report

from streamlit_pandas_profiling import st_profile_report
import os 



st.set_page_config(
        page_title="Shade ML App",
        page_icon="chart_with_upwards_trend",
        layout="wide",
    )
if os.path.exists('./dataset.csv'): 
    df = pd.read_csv('dataset.csv', index_col=None)

with st.sidebar: 
    side_img = Image.open('./assets/ai.png')
    img_1 = side_img.resize((200,200))
    st.image(img_1, caption="Automated ML App") 
    st.title("Automated ML Tasks")
    choice = st.radio("Navigation", ["Upload your Data","EDA","Create Your Model", "Download Your Model"])
    st.info("Explore your data and Build your model ")
    st.markdown('My Github  https://github.com/Ganryuu')  


if choice == "Upload your Data":
    st.title("Upload Your Dataset")
    file = st.file_uploader("")
    if file: 
        df = pd.read_csv(file, index_col=None)
        df.to_csv('dataset.csv', index=None)
        st.dataframe(df)

if choice == "EDA": 
    st.info("Lean more about EDA : https://towardsdatascience.com/exploratory-data-analysis-8fc1cb20fd15 ")

    st.title("Exploratory Data Analysis")
    profile_df = df.profile_report() 
    st_profile_report(profile_df)

if choice == "Create Your Model": 
    chosen_target = st.selectbox('Choose your Target Column ', (df.columns))
    st.info("Target Column means the Variable you want to predict ")
    # st.warning('This app can only do Classification tasks , this is why Algorithms and Evaluation metrics are for classification only , soon enough I will add an option to chose regression tasks too', icon="⚠️")
    if st.button('Classification Task'): 
        classification.setup(df, target=chosen_target, silent=True)
        setup_df = classification.pull()
        # st.dataframe(setup_df)
        best_model = classification.compare_models()
        compare_df = classification.pull()
        st.dataframe(compare_df)
        classification.save_model(best_model, 'best_model')
    elif st.button("Regression Task"): 
        regression.setup(df, target=chosen_target, silent=True)
        setup_df = regression.pull()
        # st.dataframe(setup_df)
        best_model =regression.compare_models()
        compare_df = regression.pull()
        st.dataframe(compare_df)
        regression.save_model(best_model, 'best_model')

if choice == "Download Your Model": 
    st.image("./assets/Download-amico.png")
    with open('best_performing_model.pkl', 'rb') as f: 
        st.download_button('Download Model', f, file_name="best_model.pkl")