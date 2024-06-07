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
import time

st.set_page_config(layout="wide")

hide_github_icon = """
#GithubIcon {
  visibility: hidden;
}
"""

st.title(":gray[SnowStore Portfolio Dashboard] 📊")
st.image("final_powered_by_s2_snowflake_light_transparent.png", width=400)

# Navigation sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(" ", ["Home", "Historical Trends", "Portfolio Breakdown", "Configuration"])

st.markdown(
    """
    <style>
    .st-emotion-cache-qeahdt h1 {
        color: #EEEEEE;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

all_stocks = ['SNOW', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JPM', 'BAC', 'WFC', 'C', 'GS', 'PLD', 'AMT', 'CCI', 'SPG', 'EQIX', 'XOM', 'CVX', 'COP', 'VLO', 'MPC', 'MMM', 'HON', 'GE', 'CAT', 'BA']

def generate_mock_sector_data(time_range):
    sectors = ['Technology', 'Finance', 'Real Estate', 'Energy', 'Industrials']
    base_prices = [232.97, 98.28, 163.09, 85.45, 105.43]
    
    # Adjust prices slightly based on the time range
    if time_range == 'Last 5 minutes':
        variation = np.random.uniform(-2, 2, size=len(base_prices))
    elif time_range == 'Last 15 minutes':
        variation = np.random.uniform(-3, 3, size=len(base_prices))
    elif time_range == 'Last 30 minutes':
        variation = np.random.uniform(-4, 4, size=len(base_prices))
    elif time_range == 'Last hour':
        variation = np.random.uniform(-7, 7, size=len(base_prices))
    elif time_range == 'Last day':
        variation = np.random.uniform(-13, 13, size=len(base_prices))
    
    avg_last_prices = [base + var for base, var in zip(base_prices, variation)]
    total_values = [price * 50 for price in avg_last_prices]
    
    return pd.DataFrame({
        'SECTOR': sectors,
        'AVG_LAST_PRICE': avg_last_prices,
        'TOTAL_VALUE': total_values
    })


def generate_mock_stock_data():
    tickers = {
    'Technology': ['SNOW', 'MSFT', 'GOOGL', 'AMZN', 'META'],
    'Finance': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
    'Real Estate': ['PLD', 'AMT', 'CCI', 'SPG', 'EQIX'],
    'Energy': ['XOM', 'CVX', 'COP', 'VLO', 'MPC'],
    'Industrials': ['MMM', 'HON', 'GE', 'CAT', 'BA']
    }
    sectors = sectors = list(tickers.keys())
    base_prices = [232.97, 98.28, 163.09, 85.45, 105.43]
    
    data = []
    for sector, ticker_list in tickers.items():
        base_price = base_prices[sectors.index(sector)]
        for ticker in ticker_list:
            last_price = base_price + np.random.uniform(0.93, 1.07)
            first_price = last_price / np.random.uniform(0.93, 1.07)  # Slightly adjusted first price
            percentage_change = (last_price - first_price) / first_price * 100
            total_value = last_price * 50  
        
            data.append({
                "Stock": ticker,
                "Sector": sector,
                "Current Price": last_price,
                "Percentage Change": percentage_change,
                "Total Value": total_value
            })
    
    return pd.DataFrame(data)

# Function to generate mock historical data using Geometric Brownian Motion
def generate_mock_historical_data(stock_list, start_time, end_time):
    data = []
    for stock in stock_list:
        # Generate time series data with the specified frequency
        time_range = pd.date_range(start=start_time, end=end_time)
        num_points = len(time_range)
        
        # GBM parameters
        S0 = np.random.uniform(80, 230)  # Initial price
        mu = 0.0001  # Mean return
        sigma = 0.05  # Volatility
        dt = 1 / 252  # Time step in years (daily frequency)
        
        prices = [S0]
        for _ in range(1, num_points):
            dW = np.random.normal(0, np.sqrt(dt))
            St = prices[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
            prices.append(St)
        
        for current_time, price in zip(time_range, prices):
            bid = price - np.random.uniform(0.1, 1)
            ask = price + np.random.uniform(0.1, 1)
            volume = np.random.randint(100, 1000)
            data.append({
                "STOCK": stock,
                "TIME": current_time,
                "LAST_PRICE": price,
                "BID": bid,
                "ASK": ask,
                "VOLUME": volume
            })
    
    return pd.DataFrame(data)

overview_df = generate_mock_stock_data()

################ BREAKDOWN ###############

if page == "Portfolio Breakdown":
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

    st.title(":grey[Account Breakdown]")

    # Create a time range filter
    time_range = st.selectbox("Select time range", ['Last 5 minutes', 'Last 15 minutes', 'Last 30 minutes', 'Last hour', 'Last day'])

    with st.spinner(""):
        time.sleep(0.7)
    # Generate mock data based on selected time range
    sector_performance_df = generate_mock_sector_data(time_range)

    if sector_performance_df.empty:
        st.error("No data found for sector performance.")
    else:
        # Use columns to align the charts and table
        col1, col2 = st.columns(2)

        with col1:
            st.subheader(":gray[Average Share Price by Sector]")
            sector_chart = px.bar(sector_performance_df, x='SECTOR', y='AVG_LAST_PRICE', title=f'Average Performance by Sector ({time_range})', labels={'AVG_LAST_PRICE': 'Average Last Price'}, color_discrete_sequence=['#C573FB'], text='AVG_LAST_PRICE')
            sector_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            sector_chart.update_traces(texttemplate='%{text:.2f}', textposition='outside')
            st.plotly_chart(sector_chart)
            
        with col2:
            st.subheader(":gray[Portfolio Value by Sector]")
            pie_chart = px.pie(
                sector_performance_df, 
                names='SECTOR', 
                values='TOTAL_VALUE', 
                title=f'Portfolio Value Distribution by Sector ({time_range})',
                color_discrete_sequence=['#C573FB', '#69277F', '#EEECF2']
            )
            pie_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(pie_chart)



    st.title(":grey[Stock Overview]")

    # Sector filter
    sectors = ['All'] + overview_df['Sector'].unique().tolist()
    selected_sector = st.selectbox("Filter by", sectors)

    if selected_sector != 'All':
        overview_df = overview_df[overview_df['Sector'] == selected_sector]

    # Sort the DataFrame based on user selection
    sort_column = st.selectbox("Sort by", ["Stock", "Current Price", "Percentage Change", "Total Value"])
    sort_ascending = st.radio("Sort order", ["Ascending", "Descending"]) == "Ascending"

    with st.spinner(""):
        time.sleep(0.5)
    # Sort the DataFrame
    overview_df = overview_df.sort_values(by=sort_column, ascending=sort_ascending)

    with st.expander("*Percentage change from last hour", expanded=True):
        for index, row in overview_df.iterrows():
            col1, col2, col3 = st.columns([1, 2, 2])
            with col1:
                st.write(f"**{row['Stock']}**")
            with col2:
                st.metric(label="Current Price", value=f"${row['Current Price']:,.2f}", delta=f"{row['Percentage Change']:.2f}%")
            with col3:
                st.metric(label="Total Value", value=f"${row['Total Value']:,.2f}")
            st.write("---")

############# HISTORICAL TRENDS ##############
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

    st.title(":gray[Historical Trends]")

    # Generate mock data
    selected_stocks = st.multiselect("Select Stocks for General Analysis", all_stocks, default=['VLO'])

    # Dropdown to select time range
    time_ranges = {
        "Last Day": 1440,
        "Last Week": 10080,
        "Last Month": 43200,
        "Last Year": 525600,
        "All time": 2628000
    }
    selected_range = st.selectbox("Select Time Range", list(time_ranges.keys()), index=2)

    # # Determine frequency based on selected time range
    # if selected_range in ["Last Day", "Last Week"]:
    #     freq = '1H'
    # elif selected_range == "Last Month":
    #     freq = '1D'
    # elif selected_range in ["Last Year", "All time"]:
    #     freq = '10D'
    # else:
    #     freq = '1D'  # Default frequency

    if selected_stocks and selected_range:
        selected_range_minutes = time_ranges[selected_range]
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=selected_range_minutes)
        
        with st.spinner("Loading data..."):
            time.sleep(3)  # Delay for 2 seconds
            # Generate mock historical data
        filtered_df = generate_mock_historical_data(selected_stocks, start_time, end_time)

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

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📈 Price Over Time")
            price_chart = px.line(filtered_df, x='TIME', y='LAST_PRICE', labels={'LAST_PRICE': 'Last Price'}, color='STOCK', title=f'Price Over Time ({selected_range})', color_discrete_sequence=['#29B5E8', '#11567F', '#8A999E', '#000000'])
            price_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(price_chart)

        with col2:
            st.subheader("📉 Volume Over Time")
            volume_chart = px.bar(filtered_df, x='TIME', y='VOLUME', labels={'VOLUME': 'Volume'}, color='STOCK', title=f'Volume Over Time ({selected_range})', color_discrete_sequence=['#29B5E8', '#11567F', '#8A999E', '#000000'])
            volume_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(volume_chart)

        st.write("---")
        col3, col4 = st.columns(2)
        with col3:     
            st.subheader("📊 Bid-Ask Spread Over Time")
            spread_chart = px.line(filtered_df, x='TIME', y='BID_ASK_SPREAD', color='STOCK', labels={'BID_ASK_SPREAD': 'Bid-Ask Spread'}, title=f'Bid-Ask Spread Over Time ({selected_range})', color_discrete_sequence=['#29B5E8', '#11567F', '#8A999E', '#000000'])
            spread_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(spread_chart)
        
        with col4:
            st.subheader("📈 Percentage Return Over Time")
            return_chart = px.line(filtered_df, x='TIME', y='PERCENT_RETURN', color='STOCK', labels={'PERCENT_RETURN': 'Percent Return'}, title=f'Percentage Return Over Time ({selected_range})', color_discrete_sequence=['#29B5E8', '#11567F', '#8A999E', '#000000'])
            return_chart.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(return_chart)            

    else:
        st.info("Please select at least one stock and a time range to view the historical trend.")


elif page == "Configuration":

    st.image("architecture.png")
    st.title(":gray[Configuration]")

    st.subheader(":gray[Table DDL]")
    col1, col2 = st.columns(2)

    with col1:
        st.code("""
        CREATE TABLE FINANCIAL_DATA (
            TIME datetime DEFAULT NULL,
            STOCK VARCHAR(50)
            LAST_PRICE INT(11) DEFAULT NULL,
            BID DOUBLE DEFAULT NULL,
            ASK DOUBLE DEFAULT NULL,
            VOLUME BIGINT(20) DEFAULT NULL,
            YEAR_MONTH VARCHAR(50),
            SHARD KEY (STOCK),
            SORT KEY TIME (TIME DESC)
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
    
    st.subheader(":gray[Sample Queries]")
    col1, col2 = st.columns(2)

    with col1:
        st.code("""
            -- Average performance per sector
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
        """, language='sql')

    with col2:
        st.code("""
    -- Historical portfolio values by sector
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
        """, language='sql')

else:

    # Display Global View
    st.header(":gray[Portfolio Overview]")

    account = st.selectbox("Select an account", ["Individual", "IRA", "Margin"])

    portfolio_value_default = np.random.uniform(200000, 250000)
    portfolio_change_percent_default = np.random.uniform(20, 25)
    portfolio_change_default = portfolio_value_default * (portfolio_change_percent_default / 100)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style='padding: 20px; border: 1px solid #F6EFFC; border-radius: 7px; background-color: #E1C4F7; text-align: center;'>
                <h2 style='margin: 0;'>Total Account Value</h2>
                <p style='font-size: 24px; margin: 0;'><strong>${portfolio_value_default:,.2f}</strong></p>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        change_color = "#28a745" 
        st.markdown(f"""
            <div style='padding: 20px; border: 1px solid #F6EFFC; border-radius: 5px; background-color: #E1C4F7; text-align: center;'>
                <h2 style='margin: 0;'>Total Change</h2>
                <p style='font-size: 24px; margin: 0; color: {change_color};'><strong>${portfolio_change_default:,.2f} ({portfolio_change_percent_default:.2f}%)</strong></p>
            </div>
        """, unsafe_allow_html=True)

    st.write("---")


    top_performers = overview_df.nlargest(5, 'Percentage Change').reset_index(drop=True)
    top_performers.index += 1
    bottom_performers = overview_df.nsmallest(5, 'Percentage Change').reset_index(drop=True)
    bottom_performers.index += 1

    col1, col2 = st.columns(2)  
    def highlight_positive(val):
        color = 'green' if val > 0 else 'red'
        return f'color: {color}'

    top_performers_styled = top_performers.style.format({
        'Current Price': '{:.2f}',
        'Percentage Change': '{:.2f}',
        'Total Value': '{:.2f}'
    }).applymap(highlight_positive, subset=['Percentage Change'])

    bottom_performers_styled = bottom_performers.style.format({
        'Current Price': '{:.2f}',
        'Percentage Change': '{:.2f}',
        'Total Value': '{:.2f}'
    }).applymap(highlight_positive, subset=['Percentage Change'])


    with col1:
        st.markdown("<h2 style='color: #9B3DE5;'>Top Performers</h2>", unsafe_allow_html=True)
        st.dataframe(top_performers_styled, height=200)  # Adjust height as needed

    with col2:
        st.markdown("<h2 style='color: #9B3DE5;'>Bottom Performers</h2>", unsafe_allow_html=True)
        st.dataframe(bottom_performers_styled, height=200)  # Adjust height as needed


    # def generate_mock_historical_portfolio_value(end_value):
    #     end_date = datetime.now()
    #     start_date = end_date - timedelta(days=5*365)  # 5 years ago
    #     date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    #     start_value = 150000
    #     num_days = len(date_range)
    #     drift = (end_value / start_value) ** (1 / num_days)
        
    #     # Generate a random walk for the portfolio value with Geometric Brownian Motion
    #     values = [start_value]
    #     for _ in range(1, num_days - 1):
    #         random_shock = np.random.normal(0, 0.01)
    #         next_value = values[-1] * drift * np.exp(random_shock)
    #         next_value = min(next_value, 350000)
    #         values.append(next_value)

    #     values[-1] = (values[-1] + end_value) / 2
    #     values.append(end_value)

    #     # values.append(end_value)
    #     data = {
    #         'TIME': date_range,
    #         'TOTAL_VALUE': values
    #     }
    #     return pd.DataFrame(data)

    # # Generate mock data
    # historical_total_value_df = generate_mock_historical_portfolio_value(portfolio_value_default)

    def generate_mock_historical_portfolio_value(end_value, filename):
        if os.path.exists(filename):
            # Load data from the file if it exists
            df = pd.read_csv(filename)
            df['TIME'] = pd.to_datetime(df['TIME'])  # Ensure the TIME column is in datetime format
            return df
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5*365)  # 5 years ago
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        start_value = 150000
        num_days = len(date_range)
        drift = (end_value / start_value) ** (1 / num_days)
        
        # Generate a random walk for the portfolio value with Geometric Brownian Motion
        values = [start_value]
        for i in range(1, num_days - 10):
            random_shock = np.random.normal(0, 0.01)
            next_value = values[-1] * drift * np.exp(random_shock)
            next_value = min(next_value, 350000)
            values.append(next_value)

        # Smooth transition for the last 10 days
        for i in range(num_days - 10, num_days):
            next_value = values[-1] + (end_value - values[-1]) / (num_days - i)
            values.append(next_value)

        data = {
            'TIME': date_range,
            'TOTAL_VALUE': values
        }
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)  # Save the data to a CSV file
        return df

    # Generate mock data or load from file
    filename = "historical_portfolio_value.csv"
    historical_total_value_df = generate_mock_historical_portfolio_value(portfolio_value_default, filename)
        
    st.markdown("<h2 style='color: #29B5E8;'>Historical Portfolio Value</h2>", unsafe_allow_html=True)
    total_value_chart = px.line(historical_total_value_df, x='TIME', y='TOTAL_VALUE', title='Historical Portfolio Total Value', labels={'TOTAL_VALUE': 'Total Value', 'TIME': 'Time'}, color_discrete_sequence=['#29B5E8'])
    st.plotly_chart(total_value_chart, use_container_width=True)



