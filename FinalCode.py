#Name: Rohit Sharma
#Class: CISC 599 
#Final Project
#************************************************

#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import base64


#Settings the page to create a working web-based dashboard

st.set_page_config(page_title="California State Power Balance Report", layout="wide", page_icon="⚡")

#The background image for the webpage and glass effect to the different sections that shows information
st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-image: url('https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ffeature.wecc.org%2Fwara%2Fassets%2FCIETlMdsfC%2Fgettyimages-1329323481-4096x2731.jpg&f=1&nofb=1&ipt=de287cc70f76c5fe8ee5197ff5ae79178d489fdeecd685dad20d2000a3a34f90');
            background-size: cover;
            background-position: center;
        }

        .glass-box {
            background-color: rgba(255, 255, 255, 0.5);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.5);
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

#Creating a cache memorny to store the Power generationa nd load load for the year of 2021, 2022, 2023, 2024 and 2025.
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Users\rrohi\Downloads\bigData.csv', index_col=0) #loading the csv file
    df = df.rename(columns={'Date_x': 'Date'})                             #Rename Date column as it was not in proper format
    df['Date'] = pd.to_datetime(df['Date'])                                #Extracting date
    df['Year'] = df['Date'].dt.year                                        #Extracting year
    df['Month_num'] = df['Date'].dt.month                                  #Extracting month
    df['Month'] = df['Date'].dt.month_name()                               #Creating new coulmn with Month as name
    df = df.drop(['Reason'], axis=1)                                       #Dropping unneccessary columns
    df.columns = df.columns.str.strip()                                    #Dropping extra spaces in the column names if any.
    return df

df = load_data()

#Report Title
st.markdown("""
   <h1 style='font-size: 2em; color: #228B22; text-align:center;'> Harrisburg University CISC 699</h1>   
""", unsafe_allow_html=True)
st.markdown("""
    <div class='glass-box'>
        <h1 style='font-size: 3em; color: #000000; text-align:center;'> California Power Balance Report ⚡</h1>
    </div>
""", unsafe_allow_html=True)


# Sidebar Inputs to select Year, month and attribute to predict for the 2026.
st.sidebar.title("Select Period")
selected_year = st.sidebar.selectbox("Select Year", options=["All"] + sorted(df['Year'].dropna().unique().tolist()))
selected_month = st.sidebar.selectbox("Select Month", options=["All"] + (df['Month'].dropna().unique().tolist()))
selected_source = st.sidebar.selectbox("Forecast", ["Solar", "Wind", "Nuclear", "Load","Thermal","Imports"])  


# Filtering Data as per user input
df_filtered = df.copy()
if selected_year != "All":
    df_filtered = df_filtered[df_filtered['Year'] == int(selected_year)]
if selected_month != "All":
    df_filtered = df_filtered[df_filtered['Month'] == selected_month]

if df_filtered.empty:
    st.warning("No data found for the selected period.")
    st.stop()

st.markdown(f"""
    <div class='glass-box'>
        <h2 style='color: #000000; text-align: center;'>Overall 2021 through 2025 Load and Generation</h2>
    </div>
""", unsafe_allow_html=True)

#Summary of the Generation source for the selected month and year
summary=df.copy()
summary= summary.groupby("Year")[["Net Load","Generation"]].sum().reset_index()
summary_fixed = summary.melt(id_vars="Year", value_vars=["Net Load", "Generation"],var_name="Type", value_name="Value")
fig = px.bar(summary_fixed,x="Year",y="Value",color="Type",barmode="group",title="Load and Generation 2021 through 2025",labels={"Value": "Generation (GWh)"})
st.plotly_chart(fig, use_container_width=True)

summary2=df.copy()
summary2= summary2.groupby("Year")[["Thermal","Imports","Solar","Wind","Nuclear"]].sum().reset_index()
summary2_fixed = summary2.melt(id_vars="Year", value_vars=["Thermal","Imports","Solar","Wind","Nuclear"],var_name="Type", value_name="Value")
fig1 = px.bar(summary2_fixed,x="Year",y="Value",color="Type",barmode="group",title="Generation 2021 through 2025",labels={"Value": "Generation (GWh)"})
st.plotly_chart(fig1, use_container_width=True)

st.markdown(f"""
    <div class='glass-box'>
        <h2 style='color: #000000; text-align: center;'>Summary Report - {selected_month} {selected_year}</h2>
    </div>
""", unsafe_allow_html=True)


if not df_filtered.empty:
    df_pie=df_filtered.copy()
    source_sums = df_pie[["Thermal","Imports","Solar","Wind","Nuclear"]].sum().reset_index()
    source_sums.columns = ['Source', 'Generation']
    pie_chart = px.pie(source_sums, values='Generation', names='Source', title='Generation by Source')
    st.plotly_chart(pie_chart, use_container_width=True)



#Predicting for 2026 for the same month selected by user for teh selected attribute.

st.markdown(f"""
    <div class='glass-box'>
        <h2 style='color: #000000; text-align: left;font-size:24px;'>Prediction for {selected_month} 2026 </h2>
    </div>
""", unsafe_allow_html=True)


#Grouping the data by month for prediction
df_monthly = df.groupby(['Year', 'Month_num']).agg({'Solar': 'sum', 'Wind': 'sum', 'Nuclear': 'sum','Load': 'sum','Thermal': 'sum','Imports': 'sum'}).reset_index()

#Renaming month column as prediction model needs value than text.
df_monthly['ds'] = pd.to_datetime(
    df_monthly.assign(day=1)[['Year', 'Month_num', 'day']].rename(columns={'Year':'year', 'Month_num':'month', 'day':'day'})
)

X = df_monthly[['Year', 'Month_num']]
y = df_monthly[['Solar', 'Wind', 'Nuclear','Load','Thermal','Imports']] #values to predict

#providing input to the regresssion model with 80% train data and 20% test data.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
model.fit(X_train, y_train)
#predicting for 2026 for 12 months
future = pd.DataFrame({
    'Year': [2026] * 12,
    'Month_num': list(range(1, 13))
})
predictions = model.predict(future)
df_pred = future.copy()
df_pred[['Solar', 'Wind', 'Nuclear','Load','Thermal','Imports']] = predictions




#Ploting the bar chart for entire 2026 for the predicted values
predict_plot=df_pred.copy()
predict_plot_fixed = predict_plot.melt(id_vars="Month_num", value_vars=['Solar', 'Wind', 'Nuclear','Load','Thermal','Imports'],var_name="Type", value_name="Value")
fig2 = px.bar(predict_plot_fixed,x="Month_num",y="Value",color="Type",barmode="group",title="Generation Prediction 2026",labels={"Value": "Generation (GWh)"})
st.plotly_chart(fig2, use_container_width=True)


month_map = {1: "January", 2: "February", 3: "March", 4: "April",5: "May", 6: "June", 7: "July", 8: "August",9: "September", 10: "October", 11: "November", 12: "December"}

df_predict_pie=df_pred.copy()
df_predict_pie["Month_name"] = df_predict_pie["Month_num"].map(month_map)
df_predict_pie = df_predict_pie[df_predict_pie['Month_name'] == selected_month]
source_predict_sums = df_predict_pie[["Thermal","Imports","Solar","Wind","Nuclear"]].sum().reset_index()
source_predict_sums.columns = ['Source', 'Generation']
pie_chart2 = px.pie(source_predict_sums, values='Generation', names='Source', title='Generation Prediction by Source')
st.plotly_chart(pie_chart2, use_container_width=True)



#Ploting the bar chart for selected month in  2026 for the predicted values
if selected_month != "All":
    selected_month_num = pd.to_datetime(f"2025 {selected_month} 01").month

    forecast_selected = df_pred[df_pred['Month_num'] == selected_month_num].copy()
    forecast_selected['ds'] = pd.to_datetime(dict(year=forecast_selected['Year'], month=forecast_selected['Month_num'], day=1))
    forecast_selected['yhat'] = forecast_selected[selected_source]
    forecast_selected['Time']=forecast_selected['ds']
    forecast_selected['Prediction']=forecast_selected['yhat']

    st.write(forecast_selected[['Time', 'Prediction']])
    df_monthly_source = df.groupby(['Year', 'Month_num'])[selected_source].sum().reset_index()
    df_monthly_source['ds'] = pd.to_datetime(dict(year=df_monthly_source['Year'], month=df_monthly_source['Month_num'], day=1))
    df_monthly_filtered = df_monthly_source[df_monthly_source['ds'].dt.month == selected_month_num]
    line_fig = px.line()
    line_fig.add_scatter(x=df_monthly_filtered['ds'], y=df_monthly_filtered[selected_source], mode='lines', name='Historical')
    line_fig.add_scatter(x=forecast_selected['ds'], y=forecast_selected['yhat'], mode='lines+markers', name='Forecast 2026')
    st.plotly_chart(line_fig, use_container_width=True)
   


