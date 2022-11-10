
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st

start = '2010-01-01'
end='2019-12-31'
st.title('Stock trend prediction')
user_input = st.text_input('Enter Stock Ticker','AAPL')
df=data.DataReader(user_input,'yahoo',start,end)
st.subheader('data drom 2010-2022')
st.write(df.describe())

st.subheader('Closing Prive vs time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Prive vs time Chart with 100 MA')
ma100=df.Close.rolling(100).mean()
 
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)
st.subheader('Closing Prive vs time Chart with 100 MA')
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
 
fig=plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)


model=load_model('keras_model.h5')
past_100_days=data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)

input_data=scaler.fit_transform(final_df) 

