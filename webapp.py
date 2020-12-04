import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import math

app = dash.Dash()
server = app.server
#start for apple
df=pd.read_csv("AppleData.csv")
df["Date1"]=pd.to_datetime(df.Date1,format="%m/%d/%Y")
df.index=df['Date1']
#sorting and filtering the data
data=df.sort_index(ascending=True,axis=0)
new_dataset=pd.DataFrame(index=range(0,len(df)),columns=['Date','Close'])
for i in range(0,len(data)):
    new_dataset["Date"][i]=data['Date1'][i]
    new_dataset["Close"][i]=data["Close"][i]
#normalizing the data
final_dataset=new_dataset.values
new_dataset.index=new_dataset.Date
new_dataset.drop("Date",axis=1,inplace=True)
final_dataset=new_dataset.values
value_of_len=math.ceil(len(new_dataset)*0.8)
train_data=final_dataset[0:value_of_len,:]#breaking the data into traning and validation
valid_data=final_dataset[value_of_len:,:]
scaler=MinMaxScaler(feature_range=(0,1))#creating a variable to scale the data on its features between 0 and 1
scaled_data=scaler.fit_transform(final_dataset)#transforming the data and giving it values between 0 and 1 based on its featres

a_train_data=[]
b_train_data=[]

for i in range(10,value_of_len):
    a_train_data.append(scaled_data[i-10:i,0])#will contain the first 50 vales from scaled data
    b_train_data.append(scaled_data[i,0])#will contain the 51st value

a_train_data=np.array(a_train_data)#converting the data into numpy arrays
b_train_data=np.array(b_train_data)
a_train_data=np.reshape(a_train_data,(a_train_data.shape[0],a_train_data.shape[1],1))


model=load_model("saved_model.h5")

inputs_data=new_dataset[len(new_dataset)-len(valid_data)-10:].values
inputs_data=inputs_data.reshape(-1,1)
inputs_data=scaler.transform(inputs_data)

X_test=[]
for i in range(10,inputs_data.shape[0]):
    X_test.append(inputs_data[i-10:i,0])
X_test=np.array(X_test)

X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price=model.predict(X_test)
closing_price=scaler.inverse_transform(closing_price)

train=new_dataset[:value_of_len]
valid=new_dataset[value_of_len:]
valid['Predictions']=closing_price
#end of apple
#start for facebook
df_facebook=pd.read_csv("FacebookStock.csv")
df_facebook["Date"]=pd.to_datetime(df_facebook.Date,format="%m/%d/%Y")
df_facebook.index=df_facebook['Date']

#sorting and filtering the data
data_facebook=df_facebook.sort_index(ascending=True,axis=0)
new_dataset_facebook=pd.DataFrame(index=range(0,len(df_facebook)),columns=['Date','Close'])
for i in range(0,len(data_facebook)):
    new_dataset_facebook["Date"][i]=data_facebook['Date'][i]
    new_dataset_facebook["Close"][i]=data_facebook["Close"][i]

#normalizing the data
final_dataset_facebook=new_dataset_facebook.values
new_dataset_facebook.index=new_dataset_facebook.Date
new_dataset_facebook.drop("Date",axis=1,inplace=True)
final_dataset_facebook=new_dataset_facebook.values
value_of_len_facebook=math.ceil(len(new_dataset_facebook)*0.8)
train_data_facebook=final_dataset_facebook[0:value_of_len_facebook,:]#breaking the data into traning and validation
valid_data_facebook=final_dataset_facebook[value_of_len_facebook:,:]
scaler_facebook=MinMaxScaler(feature_range=(0,1))#creating a variable to scale the data on its features between 0 and 1
scaled_data_facebook=scaler_facebook.fit_transform(final_dataset_facebook)#transforming the data and giving it values between 0 and 1 based on its featres

a_train_data_facebook=[]
b_train_data_facebook=[]

for i in range(50,value_of_len_facebook):
    a_train_data_facebook.append(scaled_data_facebook[i-50:i,0])#will contain the first 50 vales from scaled data
    b_train_data_facebook.append(scaled_data_facebook[i,0])#will contain the 51st value

a_train_data_facebook=np.array(a_train_data_facebook)#converting the data into numpy arrays
b_train_data_facebook=np.array(b_train_data_facebook)
a_train_data_facebook=np.reshape(a_train_data_facebook,(a_train_data_facebook.shape[0],a_train_data_facebook.shape[1],1))

model_facebook=load_model("Facebook_model.h5")

inputs_data_facebook=new_dataset_facebook[len(new_dataset_facebook)-len(valid_data_facebook)-10:].values
inputs_data_facebook=inputs_data_facebook.reshape(-1,1)
inputs_data_facebook=scaler_facebook.transform(inputs_data_facebook)

X_test_facebook=[]
for i in range(10,inputs_data_facebook.shape[0]):
    X_test_facebook.append(inputs_data_facebook[i-10:i,0])
X_test_facebook=np.array(X_test_facebook)

X_test_facebook=np.reshape(X_test_facebook,(X_test_facebook.shape[0],X_test_facebook.shape[1],1))
closing_price_facebook=model_facebook.predict(X_test_facebook)
closing_price_facebook=scaler_facebook.inverse_transform(closing_price_facebook)

train_facebook=new_dataset_facebook[:value_of_len_facebook]
valid_facebook=new_dataset_facebook[value_of_len_facebook:]
valid_facebook['Predictions']=closing_price_facebook

#end of facebook
app.layout = html.Div([
   
    html.H1("Stock Price Predictor ", style={"textAlign": "center"}),
    dcc.Tabs(id="tabs", children=[
            dcc.Tab(label='Apple',children=[
                    html.Div([
                        html.H2("Actual closing price",style={"textAlign": "center"}),
                        dcc.Graph(
                            id="Actual Data",
                            figure={
                                "data":[
                                    go.Scatter(
                                        x=valid.index,
                                        y=valid["Close"],
                                        mode='markers'
                                    )
                                ],
                                "layout":go.Layout(
                                    title='scatter plot',
                                    xaxis={'title':'Date'},
                                    yaxis={'title':'Closing Rate'}
                                )
                            }
                        ),
                        html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                        dcc.Graph(
                            id="Predicted Data",
                            figure={
                                "data":[
                                    go.Scatter(
                                        x=valid.index,
                                        y=valid["Predictions"],
                                        mode='markers'
                                    )
                                ],
                                "layout":go.Layout(
                                    title='scatter plot',
                                    xaxis={'title':'Date'},
                                    yaxis={'title':'Closing Rate'}
                                )
                            }
                        )                
                    ])                
                ]),
            dcc.Tab(label='Facebook',children=[
                 html.Div([
                     html.H2("Actual closing price",style={"textAlign": "center"}),
                     dcc.Graph(
                            id="Actual Data facebook",
                            figure={
                                "data":[
                                    go.Scatter(
                                        x=valid_facebook.index,
                                        y=valid_facebook["Close"],
                                        mode='markers'
                                    )
                                ],
                                "layout":go.Layout(
                                    title='scatter plot',
                                    xaxis={'title':'Date'},
                                    yaxis={'title':'Closing Rate'}
                                )
                            }
                        ),
                        html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                        dcc.Graph(
                            id="Predicted Data facebook",
                            figure={
                                "data":[
                                    go.Scatter(
                                        x=valid_facebook.index,
                                        y=valid_facebook["Predictions"],
                                        mode='markers'
                                    )
                                ],
                                "layout":go.Layout(
                                    title='scatter plot',
                                    xaxis={'title':'Date'},
                                    yaxis={'title':'Closing Rate'}
                                )
                            }
                        )
                 ])
            ])
    ])
])

if __name__=='__main__':
    app.run_server(debug=True)
