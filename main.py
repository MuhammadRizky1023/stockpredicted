from datetime import date
import streamlit as st
from plotly import graph_objects as go
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Stock Price Prediction by Region")

# Define stocks by region
stocks_by_region = {
    "America": ["AAPL", "GOOG", "MSFT", "GME"],
    "Europe": ["VOW3.DE", "BMW.DE", "DTE.DE", "OR.PA", "AIR.PA", "MC.PA", "HSBA.L", "BP.L", "AZN.L"],
    "Asia": ["7203.T", "6758.T", "9984.T", "005930.KS", "000660.KS", "BABA", "0700.HK", "3690.HK"], 
    "Southeast Asia": ["BBCA.JK", "TLKM.JK", "1155.KL", "4863.KL", "D05.SI", "C07.SI", "PTT.BK", "SCC.BK", "AC.PS", "SM.PS"]
}

region = st.selectbox("Select region for prediction", list(stocks_by_region.keys()))
selected_stock = st.selectbox("Select dataset for prediction", stocks_by_region[region])

n_years = st.slider("Years of prediction:", 1, 10)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Loading data...")
data = load_data(selected_stock)
data_load_state.text("Loading data...done!")
st.header("Raw data")
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.write("Forecast plot")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
