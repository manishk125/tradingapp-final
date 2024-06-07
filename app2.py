import streamlit as st
import datetime
import snowflake.connector
from snowflake.snowpark import Session
from snowflake.snowpark import functions as f
# import spcs_helpers
import singlestoredb as s2
import pandas as pd
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(layout="wide")


st.title("SnowStore Portfolio Dashboard 📊")
st.image("logo.png", width=400)

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select", ["Home", "Portfolio Overview", "Historical Trends", "Configuration"])

sidebar_title_color = """
<style>
    .st-emotion-cache-dvne4q h2 {  # This class name may change, inspect your sidebar title using browser dev tools to get the exact class name
        color: #F2F2F2;  # Replace with your desired color
    }
</style>
"""

# Apply the custom CSS
st.markdown(sidebar_title_color, unsafe_allow_html=True)

st.markdown(
    f"""
    <style>
    .st-emotion-cache-1gv3huu {{
        background-color: #360061;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

stock_list = ['META', 'SMCI', 'GS', 'SNOW', 'CRM', 'QCOM', 'AMZN', 'AMD', 'TSLA', 'DELL', 'MSFT', 'TSM', 'HOOD', 'NVDA', 'ASML']

numshares = 50

# S2 config 
# conn_params = {
#   'host': 'svc-89083124-c419-44df-b3ba-425e1b595432-dml.aws-virginia-6.svc.singlestore.com',
#   'user': 'admin', 
#   'password': 'SingleStore1234!', 
#   'database': 'vectordb',
#   'port': 3306, 
# }
conn_params = {
  'host': 'svc-bd788f8f-c95e-470d-a767-462840795288-dml.aws-oregon-3.svc.singlestore.com',
  'user': 'admin', 
  'password': 'Singlestore1', 
  'database': 'vectordb',
  'port': 3306, 
}

def connect_to_s2():
    try:
        return s2.connect(**conn_params)
    except s2.Error as e:
        st.error(f"Error connecting to SingleStore: {e}")
        return None
    # return s2.connect(**st.secrets["singlestore"])

# conn = connect_to_s2()

def connect_to_snowflake():
    try:
        conn = snowflake.connector.connect(
            account = 'SOUQODV-SNOWFLAKE_INTEGRATION',
            user = 'INGRIDXU',
            password = 'ukt-pju7tjq7quj4VFZ',
            warehouse = 'CONTAINER_HOL_WH',
            role = 'ACCOUNTADMIN',
            database='stock_db',
            schema='PUBLIC'
        )
        return conn
    except snowflake.connector.Error as e:
        st.error(f"Error connecting to Snowflake: {e}")
        return None


def run_query(conn, query, params=None):
    try:
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return pd.DataFrame(result, columns=columns)
    except s2.Error as e:
        st.error(f"Error running query: {e}")
        return pd.DataFrame()

base_query = """
SELECT * FROM FINANCIAL_DATA ORDER BY TIME DESC;
"""

base_query_enriched = """
SELECT t.TIME, t.STOCK, t.LAST_PRICE, t.BID, t.ASK, t.VOLUME, e.COMPANY_NAME, e.COUNTRY, e.MARKET_CAP, e.IPO_YEAR, e.SECTOR, e.INDUSTRY
FROM FINANCIAL_DATA t
JOIN FINANCIAL_ENRICHMENT e ON t.STOCK = e.STOCK
"""

historical_total_value_query = """
SELECT 
    TIME,
    SUM(LAST_PRICE * 50) AS TOTAL_VALUE
FROM 
    FINANCIAL_DATA
GROUP BY 
    TIME
ORDER BY 
    TIME ASC;
"""

# selected_stocks_query_enriched = f"""
# SELECT t.TIME, t.STOCK, t.LAST_PRICE, t.BID, t.ASK, t.VOLUME, e.COMPANY_NAME, e.COUNTRY, e.MARKET_CAP, e.IPO_YEAR, e.SECTOR, e.INDUSTRY
# FROM FINANCIAL_DATA t
# JOIN FINANCIAL_ENRICHMENT e ON t.STOCK = e.STOCK
# WHERE t.STOCK IN ({', '.join(['%s'] * len(selected_stocks))})
# AND t.TIME BETWEEN %s AND %s
# ORDER BY t.TIME DESC
# """

# fetch performance by sector
sector_performance_query = """
SELECT 
    e.SECTOR,
    AVG(f.LAST_PRICE) AS AVG_LAST_PRICE,
    AVG(f.VOLUME) AS AVG_VOLUME,
    SUM(f.VOLUME) AS TOTAL_VOLUME,
    SUM(f.LAST_PRICE * 50) AS TOTAL_VALUE
FROM 
    FINANCIAL_DATA f
JOIN 
    FINANCIAL_ENRICHMENT e ON f.STOCK = e.STOCK
WHERE 
    f.TIME BETWEEN %s AND %s
GROUP BY 
    e.SECTOR
ORDER BY 
    AVG_LAST_PRICE DESC;
"""

# fetch historical portfolio values by sector
historical_sector_performance_query = """
SELECT 
    t.TIME,
    e.SECTOR,
    SUM(t.LAST_PRICE * 50) AS TOTAL_VALUE
FROM 
    FINANCIAL_DATA t
JOIN 
    FINANCIAL_ENRICHMENT e ON t.STOCK = e.STOCK
WHERE 
    t.TIME BETWEEN %s AND %s
GROUP BY 
    t.TIME, e.SECTOR
ORDER BY 
    t.TIME ASC;
"""


def load_data(query, params=None, db='s2'):
    if db == 'snowflake':
        conn = connect_to_snowflake()
    else:
        conn = connect_to_s2()
    if conn is None:
        st.stop()
    data = run_query(conn, query, params)
    conn.close()
    return data

# Load all data
data = load_data(base_query)

# Convert time column to datetime
data['TIME'] = pd.to_datetime(data['TIME'])
end_time = data['TIME'].max()
start_time_default = end_time - timedelta(minutes=525600)  # Default to last year

# Convert datetime to string for SQL query
start_time_str_default = start_time_default.strftime('%Y-%m-%d %H:%M:%S')
end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

# Function to calculate portfolio values and percentage change
def calculate_portfolio_values(filtered_df):
    latest_prices = filtered_df.groupby('STOCK')['LAST_PRICE'].last()
    initial_prices = filtered_df.groupby('STOCK')['LAST_PRICE'].first()
    portfolio_value = latest_prices.sum() * numshares
    initial_portfolio_value = initial_prices.sum() * numshares
    portfolio_change = portfolio_value - initial_portfolio_value
    portfolio_change_percent = (portfolio_change / initial_portfolio_value) * 100
    top_performers = latest_prices.sort_values(ascending=False).head(5)
    bottom_performers = latest_prices.sort_values(ascending=True).head(5)
    return portfolio_value, portfolio_change, portfolio_change_percent, top_performers, bottom_performers

if page == "Portfolio Overview":
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: #EBD8F9;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("Account Overview")
    
    # Dropdown to select time range
    time_ranges = {
        "Last Minute": 1,
        "Last 5 Minutes": 5,
        "Last 15 Minutes": 15,
        "Last Hour": 60,
        "Last Day": 1440
    }
    selected_range = st.selectbox("Select Time Range", list(time_ranges.keys()), index=3)
    st.write("---")
    selected_range_minutes = time_ranges[selected_range]

    # db = 's2' if selected_range_minutes <= 60 else 'snowflake'
    db = 's2'

    # Update start and end time based on selected range
    end_time = data['TIME'].max()
    start_time = end_time - timedelta(minutes=selected_range_minutes)
    start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
    end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')

    sectors = load_data("SELECT DISTINCT SECTOR FROM FINANCIAL_ENRICHMENT;", db=db)['SECTOR'].tolist()
    selected_sectors = st.multiselect("Select Sectors", sectors, default='Technology')

    sector_filter = " AND e.SECTOR IN ({})".format(", ".join(["%s"] * len(selected_sectors)))
    sector_performance_query_filtered = sector_performance_query.replace("ORDER BY", f"{sector_filter} ORDER BY")

    # Sector performance analysis
    params = [start_time_str, end_time_str] + selected_sectors
    sector_performance_df = load_data(sector_performance_query_filtered, params, db=db)
    
    if sector_performance_df.empty:
        st.error("No data found for sector performance.")
    else:
        # Use columns to align the charts and table
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Current Average Share Price by Sector")
            sector_chart = px.bar(sector_performance_df, x='SECTOR', y='AVG_LAST_PRICE', title='Average Performance by Sector', labels={'AVG_LAST_PRICE': 'Average Last Price'}, color_discrete_sequence=['#C573FB'])
            sector_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(sector_chart)
            
            # st.subheader("Portfolio Value per Sector")
            # historical_sector_performance_df = load_data(historical_sector_performance_query, params)
            
            # if not historical_sector_performance_df.empty:
            #     historical_sector_performance_df['TIME'] = pd.to_datetime(historical_sector_performance_df['TIME'])
                
            #     sector_value_chart = px.line(historical_sector_performance_df, 
            #                                  x='TIME', 
            #                                  y='TOTAL_VALUE', 
            #                                  color='SECTOR', 
            #                                  title='Historical Portfolio Value per Sector',
            #                                  color_discrete_sequence=['#69277F', '#69277F', '#C573FB'],
            #                                  labels={'TOTAL_VALUE': 'Total Value', 'TIME': 'Time', 'SECTOR': 'Sector'})
            #     sector_value_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            #     st.plotly_chart(sector_value_chart)
            # else:
            #     st.error("No historical data found for the selected criteria.")

        with col2:
            st.subheader("Portfolio Value by Sector")
            pie_chart = px.pie(
                sector_performance_df, 
                names='SECTOR', 
                values='TOTAL_VALUE', 
                title='Portfolio Value Distribution by Sector',
                color_discrete_sequence=['#C573FB', '#69277F', '#EEECF2']
            )
            pie_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(pie_chart)

            # st.subheader("Total Portfolio Value by Sector")
            # sector_value_table = sector_performance_df[['SECTOR', 'TOTAL_VALUE']]
            # sector_value_table.columns = ['Sector', 'Total Value']
            # sector_value_table['Total Value'] = sector_value_table['Total Value'].apply(lambda x: f"{x:,.2f}")
            # st.table(sector_value_table)

########################## candlestick
    # Generate fake stock data for one day
    np.random.seed(42)
    date_range = pd.date_range(start="2024-04-27 09:30", end="2024-04-27 16:00", freq="5T")
    num_points = len(date_range)

    # Create a DataFrame with random data
    data = {
        "Open": np.random.uniform(8150, 8250, num_points),
        "High": np.random.uniform(8150, 8250, num_points),
        "Low": np.random.uniform(8150, 8250, num_points),
        "Close": np.random.uniform(8150, 8250, num_points)
    }

    # Make sure high is always greater than or equal to low
    data["High"] = np.maximum(data["Open"], data["High"])
    data["Low"] = np.minimum(data["Open"], data["Low"])
    # Ensure that open and close are within the high-low range
    data["Close"] = np.clip(data["Close"], data["Low"], data["High"])

    stock_data = pd.DataFrame(data, index=date_range)

    ##### Historical Data Graph #####

    # Add a title to the historical data graph
    st.subheader("📈 Price Over Time")

    y_min = stock_data[['Low', 'Close']].min().min() - 50
    y_max = stock_data[['High', 'Close']].max().max() + 50

    # Create a plot for the historical data
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=stock_data.index,
                open=stock_data["Open"],
                high=stock_data["High"],
                low=stock_data["Low"],
                close=stock_data["Close"],
            )
        ]
    )

    # Customize the historical data graph
    fig.update_layout(xaxis_rangeslider_visible=False, title="Price Over Time (Candlestick)", yaxis=dict(range=[y_min, y_max]), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width=600)

    # Display the plot
    st.plotly_chart(fig)

########################candlestick end
    
    st.title("Stock Overview")

    # Sort the DataFrame based on user selection
    sort_column = st.selectbox("Sort by", ["Stock", "Current Price", "Percentage Change", "Total Value"])
    sort_ascending = st.radio("Sort order", ["Ascending", "Descending"]) == "Ascending"
    sort_order = "ASC" if sort_ascending else "DESC"

    # Query to fetch sorted data
    sorted_query = f"""
    SELECT
        STOCK,
        LAST_PRICE AS "Current Price",
        (LAST_PRICE - FIRST_VALUE(LAST_PRICE) OVER (PARTITION BY STOCK ORDER BY TIME)) / FIRST_VALUE(LAST_PRICE) OVER (PARTITION BY STOCK ORDER BY TIME) * 100 AS "Percentage Change",
        LAST_PRICE * {numshares} AS "Total Value"
    FROM
        FINANCIAL_DATA
    WHERE
        TIME BETWEEN %s AND %s
    ORDER BY
        "{sort_column}" {sort_order}
    """
    params = [start_time_str, end_time_str]
    overview_df = load_data(sorted_query, params, db=db)

    with st.expander("Stock Overview Details", expanded=False):
        for index, row in overview_df.iterrows():
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.write(f"**{row['STOCK']}**")
            with col2:
                st.metric(label="Current Price", value=f"${row['Current Price']:,.2f}", delta=f"{row['Percentage Change']:.2f}%")
            with col3:
                st.metric(label="Total Value", value=f"${row['Total Value']:,.2f}")
            st.write("---") 

elif page == "Historical Trends":
    st.markdown(
            f"""
            <style>
            .stApp {{
                background-color: #D5EFFA;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

    st.title("Historical Trends")
    
    selected_stocks = st.multiselect("Select Stocks for General Analysis", stock_list, default='SNOW')

    # Dropdown to select time range
    time_ranges = {
        "Last Day": 1440,
        "Last Week": 10080,
        "Last Month": 43200,
        "Last Year": 525600,
        "All time": 2628000
    }
    selected_range = st.selectbox("Select Time Range", list(time_ranges.keys()), index=3)
    st.write("---")
    if selected_stocks and selected_range:
        selected_range_minutes = time_ranges[selected_range]

        # Update start and end time based on selected range
        start_time = end_time - timedelta(minutes=selected_range_minutes)
        start_time_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        end_time_str = end_time.strftime('%Y-%m-%d %H:%M:%S')
        params = selected_stocks + [start_time_str, end_time_str]

        # selected stocks
        selected_stocks_query = f"""
        SELECT *
        FROM FINANCIAL_DATA
        WHERE STOCK IN ({', '.join(['%s'] * len(selected_stocks))})
        AND TIME BETWEEN %s AND %s
        ORDER BY TIME DESC
        """

        filtered_df = load_data(selected_stocks_query, params, db='snowflake')

        if filtered_df.empty:
            st.error("No data found for the selected criteria.")
            st.stop()

    # General Analysis
        filtered_df['TIME'] = pd.to_datetime(filtered_df['TIME'])
        # Calculate bid ask spread
        filtered_df['BID_ASK_SPREAD'] = filtered_df['ASK'] - filtered_df['BID']
        # Calculate percentage return
        initial_prices = filtered_df.groupby('STOCK')['LAST_PRICE'].transform('first')
        filtered_df['PERCENT_RETURN'] = (filtered_df['LAST_PRICE'] - initial_prices) / initial_prices * 100
        # Apply a rolling average to smooth out the data
        filtered_df['SMOOTHED_LAST_PRICE'] = filtered_df.groupby('STOCK')['LAST_PRICE'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        filtered_df['VOLUME_SMOOTHED'] = filtered_df.groupby('STOCK')['VOLUME'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        filtered_df['BID_ASK_SPREAD_SMOOTHED'] = filtered_df.groupby('STOCK')['BID_ASK_SPREAD'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())
        filtered_df['PERCENT_RETURN_SMOOTHED'] = filtered_df.groupby('STOCK')['PERCENT_RETURN'].transform(lambda x: x.rolling(window=20, min_periods=1).mean())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Price Over Time")
            price_chart = px.line(filtered_df, x='TIME', y='SMOOTHED_LAST_PRICE', labels={'SMOOTHED_LAST_PRICE': 'Last Price'}, color='STOCK', title=f'Price Over Time ({selected_range})', color_discrete_sequence=['#29B5E8'])
            price_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(price_chart)

                    
        # st.subheader("📈 Price Over Time (Candlestick)")
        # fig = go.Figure(
        #     data=[
        #         go.Candlestick(
        #             x=filtered_df['TIME'],
        #             open=filtered_df['BID'],
        #             high=filtered_df['ASK'],
        #             low=filtered_df['BID'],
        #             close=filtered_df['LAST_PRICE'],
        #             name='Candlestick'
        #         )
        #     ]
        # )
        # fig.update_layout(title=f'Price Over Time ({selected_range})', yaxis_title='Price', xaxis_rangeslider_visible=False)
        # st.plotly_chart(fig)


        with col2:
            st.subheader("📉 Volume Over Time")
            volume_chart = px.bar(filtered_df, x='TIME', y='VOLUME_SMOOTHED', labels={'VOLUME_SMOOTHED': 'Volume'}, color='STOCK', title=f'Volume Over Time ({selected_range})', color_discrete_sequence=['#29B5E8'])
            volume_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(volume_chart)

        st.write("---")
        col3, col4 = st.columns(2)
        with col3:     
            st.subheader("📊 Bid-Ask Spread Over Time")
            spread_chart = px.line(filtered_df, x='TIME', y='BID_ASK_SPREAD_SMOOTHED', color='STOCK', labels={'BID_ASK_SPREAD_SMOOTHED': 'Bid-Ask Spread'}, title=f'Bid-Ask Spread Over Time ({selected_range})', color_discrete_sequence=['#29B5E8'])
            spread_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(spread_chart)
        
        with col4:
            st.subheader("📈 Percentage Return Over Time")
            return_chart = px.line(filtered_df, x='TIME', y='PERCENT_RETURN_SMOOTHED', color='STOCK', labels={'PERCENT_RETURN_SMOOTHED': 'Percent Return'}, title=f'Percentage Return Over Time ({selected_range})', color_discrete_sequence=['#29B5E8'])
            return_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(return_chart)            

    else:
        st.info("Please select at least one stock and a time range to view the historical trend.")


elif page == "Configuration":
    st.title("Configuration")

    st.subheader("Table DDL")
    col1, col2 = st.columns(2)

    with col1:
        st.code("""
        CREATE TABLE FINANCIAL_DATA (
            TIME TIMESTAMP,
            STOCK VARCHAR(50),
            LAST_PRICE FLOAT,
            BID FLOAT,
            ASK FLOAT,
            VOLUME INT
        );
        """, language='sql')

    with col2:
        st.code("""
        CREATE TABLE FINANCIAL_ENRICHMENT (
            STOCK VARCHAR(50),
            COMPANY_NAME VARCHAR(255),
            COUNTRY VARCHAR(50),
            MARKET_CAP FLOAT,
            IPO_YEAR INT,
            SECTOR VARCHAR(50),
            INDUSTRY VARCHAR(50)
        );
        """, language='sql')
    
    st.subheader("Sample Queries")
    col1, col2 = st.columns(2)

    with col1:
        st.code("""
            -- Query to fetch data for selected stocks
    SELECT *
    FROM FINANCIAL_DATA
    WHERE STOCK IN ('AAPL', 'MSFT')
    AND TIME BETWEEN '2023-01-01' AND '2023-12-31'
    ORDER BY TIME DESC;
        """, language='sql')

    with col2:
        st.code("""
    -- Query to join financial data with enrichment data
    SELECT t.TIME, t.STOCK, t.LAST_PRICE, t.BID, t.ASK, t.VOLUME, e.COMPANY_NAME, e.COUNTRY, e.MARKET_CAP, e.IPO_YEAR, e.SECTOR, e.INDUSTRY
    FROM FINANCIAL_DATA t
    JOIN FINANCIAL_ENRICHMENT e ON t.STOCK = e.STOCK
    WHERE t.TIME BETWEEN '2023-01-01' AND '2023-12-31'
    ORDER BY t.TIME DESC;
        """, language='sql')

else:

    portfolio_value_default, portfolio_change_default, portfolio_change_percent_default, top_performers, bottom_performers = calculate_portfolio_values(data)
    historical_total_value_df = load_data(historical_total_value_query)

    # Display Global View
    st.header("Portfolio Overview")

    account = st.selectbox("Select an account", ["Individual", "IRA", "Margin"])

    col1, col2 = st.columns(2)
    # col1.metric("Total Portfolio Value", f"${portfolio_value_default:,.2f}")
    # col2.metric("Total Change", f"${portfolio_change_default:,.2f}", f"{portfolio_change_percent_default:.2f}%")
    with col1:
        st.markdown(f"""
            <div style='padding: 20px; border: 1px solid #F6EFFC; border-radius: 7px; background-color: #E1C4F7; text-align: center;'>
                <h2 style='margin: 0;'>Total Account Value</h2>
                <p style='font-size: 24px; margin: 0;'><strong>${portfolio_value_default:,.2f}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        change_color = "#28a745" if portfolio_change_default >= 0 else "#dc3545"
        st.markdown(f"""
            <div style='padding: 20px; border: 1px solid #F6EFFC; border-radius: 5px; background-color: #E1C4F7; text-align: center;'>
                <h2 style='margin: 0;'>Total Change</h2>
                <p style='font-size: 24px; margin: 0; color: {change_color};'><strong>${portfolio_change_default:,.2f} ({portfolio_change_percent_default:.2f}%)</strong></p>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")

    # Top and bottom performers
    col1, col2, col3 = st.columns([1, 2, 1])  # Adjust the column widths to control table width

    with col1:
        st.markdown("<h2 style='color: #9B3DE5;'>Top Performers</h2>", unsafe_allow_html=True)
        st.table(top_performers)

    with col2:
        # Create two columns inside the middle column for top and bottom performers
        inner_col1, inner_col2 = st.columns(2)
        with inner_col1:
            st.write("")

        with inner_col2:
            st.markdown("<h2 style='color: #9B3DE5;'>Bottom Performers</h2>", unsafe_allow_html=True)
            st.table(bottom_performers)

    with col3:
        st.write("")  # Add some space to the right
    
    st.markdown("<h2 style='color: #29B5E8;'>Historical Portfolio Value</h2>", unsafe_allow_html=True)
    total_value_chart = px.line(historical_total_value_df, x='TIME', y='TOTAL_VALUE', title='Historical Portfolio Total Value', labels={'TOTAL_VALUE': 'Total Value', 'TIME': 'Time'}, color_discrete_sequence=['#29B5E8'])
    st.plotly_chart(total_value_chart, use_container_width=True)

