import streamlit as st
import pandas as pd
import matplotlib as plt
import joblib
import warnings
warnings.filterwarnings('ignore')
import plotly as px

# st.title('ADVERT AND SALES')
# st.subheader('Built by KufreKingOfficial')

data = pd.read_csv('AdvertAndSales.csv')

st.markdown("<h1 style = 'color: #DD5746; text-align: center; font-size: 60px; font-family: Monospace'>ADVERT SALES PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #FFC470; text-align: center; font-family: Serif '>Built by KufreKing</h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)

st.image('pngwing.com (9).png')
st.divider()

st.markdown("<h2 style = 'color: #F7C566; text-align: center; font-family: montserrat '>Background Of Study</h2>", unsafe_allow_html = True)
st.markdown('Develop an app that predicts sales performance based on ad spend, audience engagement, and market trends, helping businesses optimize ad budgets and maximize revenue.')

st.divider()

st.dataframe(data, use_container_width= True)

st.sidebar.image('pngwing.com (1).png', caption= 'Welcome User')

tv = st.sidebar.number_input('Television advert exp', min_value=0.0, max_value=10000.0, value=data.TV.median())
radio = st.sidebar.number_input('Radio advert exp', min_value=0.0, max_value=10000.0, value=data.Radio.median())
socials = st.sidebar.number_input('Social media exp', min_value= 0.0, max_value = 10000.0, value=data['Social Media'].median())
infl = st.sidebar.selectbox('Type of Influencer', data.Influencer.unique(), index=1)
        

#User  input
inputs = {
    'TV': [tv],
    'Radio' : [radio],
    'Social Media' : [socials],
    'Influencer' : [infl]
}

inputVar = pd.DataFrame(inputs)
st.divider()
st.header('User Input')
st.dataframe(inputVar)

#transform the user inputs. import the transformers
tv_scaler = joblib.load('TV_scaler.pk1')
radio_scaler = joblib.load('Radio_scaler.pk1')
social_scaler = joblib.load('Social Media_scaler.pk1')
influencer_encoder = joblib.load('Influencer_encoder.pk1')

#Use the imported transformers to transform the user input
inputVar['TV'] = tv_scaler.transform(inputVar[['TV']])
inputVar['Radio'] = radio_scaler.transform(inputVar[['Radio']])
inputVar['Social Media'] = social_scaler.transform(inputVar[['Social Media']])
inputVar['Influencer'] = influencer_encoder.transform(inputVar[['Influencer']])

#Bringin in the model
model = joblib.load('advertmodel.pk1')

predictbutton = st.button('Click to Predict the Sales')

if predictbutton:
    predicted = model.predict(inputVar)
    st.success(f'The Predicted Sales value is: {predicted}')
