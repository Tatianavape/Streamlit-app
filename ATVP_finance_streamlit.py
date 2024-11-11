import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

#Individual assignment for Angie Tatiana Vargas Perea - MBD 2024

#==============================================================================

import requests
import urllib

class YFinance:
    user_agent_key = "User-Agent"
    user_agent_value = ("Mozilla/5.0 (Windows NT 6.1; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/58.0.3029.110 Safari/537.36")
    
    def __init__(self, ticker):
        self.yahoo_ticker = ticker

    def __str__(self):
        return self.yahoo_ticker

    def _get_yahoo_cookie(self):
        cookie = None

        headers = {self.user_agent_key: self.user_agent_value}
        response = requests.get("https://fc.yahoo.com",
                                headers=headers,
                                allow_redirects=True)

        if not response.cookies:
            raise Exception("Failed to obtain Yahoo auth cookie.")

        cookie = list(response.cookies)[0]

        return cookie

    def _get_yahoo_crumb(self, cookie):
        crumb = None

        headers = {self.user_agent_key: self.user_agent_value}

        crumb_response = requests.get(
            "https://query1.finance.yahoo.com/v1/test/getcrumb",
            headers=headers,
            cookies={cookie.name: cookie.value},
            allow_redirects=True,
        )
        crumb = crumb_response.text

        if crumb is None:
            raise Exception("Failed to retrieve Yahoo crumb.")

        return crumb

    @property
    def info(self):
        # Yahoo modules doc informations :
        # https://cryptocointracker.com/yahoo-finance/yahoo-finance-api
        cookie = self._get_yahoo_cookie()
        crumb = self._get_yahoo_crumb(cookie)
        info = {}
        ret = {}

        headers = {self.user_agent_key: self.user_agent_value}

        yahoo_modules = ("assetProfile,"  # longBusinessSummary
                         "summaryDetail,"
                         "financialData,"
                         "majorHoldersBreakdown,"
                         "indexTrend,"
                         "defaultKeyStatistics,"
                         "majorHoldersBreakdown,"
                         "insiderHolders")

        url = ("https://query1.finance.yahoo.com/v10/finance/"
               f"quoteSummary/{self.yahoo_ticker}"
               f"?modules={urllib.parse.quote_plus(yahoo_modules)}"
               f"&ssl=true&crumb={urllib.parse.quote_plus(crumb)}")

        info_response = requests.get(url,
                                     headers=headers,
                                     cookies={cookie.name: cookie.value},
                                     allow_redirects=True)

        info = info_response.json()
        info = info['quoteSummary']['result'][0]

        for mainKeys in info.keys():
            for key in info[mainKeys].keys():
                if isinstance(info[mainKeys][key], dict):
                    try:
                        ret[key] = info[mainKeys][key]['raw']
                    except (KeyError, TypeError):
                        pass
                else:
                    ret[key] = info[mainKeys][key]

        return ret

#==============================================================================



st.set_page_config(page_title="Financial Data for S&P 500", layout="wide")

# Get the list of stock from S&P500

@st.cache_data
def get_sp500_data():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url)
    return table[0]

sp500_table = get_sp500_data()


# Get the list of stock tickers from S&P500

company_names = sp500_table['Security'].tolist()
sp500_tickers = sp500_table['Symbol'].tolist()

@st.cache_data
def get_stock_history(ticker, period):
            stock = yf.Ticker(ticker)
            return stock.history(period=period)

@st.cache_data
def get_stock_info(selected_ticker):
            stock = yf.Ticker(selected_ticker)
            info = stock.info
            major_shareholders = stock.institutional_holders
            return info, major_shareholders


# Style
st.markdown("""
<style>
    .header {
        font-size: 36px; 
        font-weight: bold;  
        color: black;  
        text-align: center;  
        font-family: 'Arial', sans-serif;  
        padding: 10px;
        background-color: white;  
        border-bottom: 4px solid #800020;  
    }
    .table-header {
        background-color: #343434; 
        color: white;            
    }
    .table th {
        background-color: #343434; 
        color: white;              
    }
    .table td {
        color: black;             
    }
</style>
""", unsafe_allow_html=True)

def format_dollar(value):
            if value is None or value == 'N/A':
                return 'N/A'
            else:
                return f"${value:,.2f}"


# Dashboard title 
col1, col2 = st.columns([7,1])
col1.markdown('<div class="header">Financial data for S&P 500</div>', unsafe_allow_html=True)
with col2:
    st.write("Data source:")
    st.image('Yahoo_Finance.png', width=100)
    

st.sidebar.markdown("""
    <style>
    .sidebar-title {
        font-weight: bold;
        color: #970715;
    }
    </style>
    """, unsafe_allow_html=True)

st.sidebar.markdown('<p class="sidebar-title">Select a tab</p>', unsafe_allow_html=True)

page = st.sidebar.selectbox("Tab", ["Summary", "Chart", 'Financials', 'Monte Carlo simulation','Relative Performance Analysis'])

st.sidebar.markdown("## S&P500 Stock Analysis Tool")
st.sidebar.markdown("Use this tool to visualize historical stock data, Financial statement and Monte carlo Simulation")
st.sidebar.markdown("### Instructions")
st.sidebar.markdown("""
1. Select a company from the dropdown.
2. Use the buttons to pick a time period for the chart.
""")

if st.button("Update Data"):
    st.write(f"Updating data for {selected_ticker}...")
    data = get_stock_data(selected_ticker)
    st.write("Data updated successfully!")

    # Mostrar los datos en un dataframe
    st.dataframe(data)

    # Crear un botón para descargar los datos en formato CSV
    csv = data.to_csv(index=True)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"{selected_ticker}_data.csv",
        mime="text/csv"
    )

# **Page 1: Summary**
if page == "Summary":
   
   
    st.markdown('<h1 style="font-size: 24px; text-decoration: underline;">Financial info</h1>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 2])  
    with col1:
    # Company selection filter
        selected_company = st.selectbox("Select a company", company_names)

    # Get the ticker of the selected company
        selected_ticker = sp500_table.loc[sp500_table['Security'] == selected_company, 'Symbol'].values[0]
     

        info, major_shareholders = get_stock_info(selected_ticker)

        # Summarize profile information
        profile_summary = {
            'Ticker': selected_ticker,
            'Company Name': info.get('longName', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Employees': f"{info.get('fullTimeEmployees', 'N/A'):,}",
            'Previous Close':format_dollar(info.get('previousClose', 'N/A')),
            'Open':format_dollar(info.get('open', 'N/A')),
            'Volume': f"{info.get('volume', 'N/A'):,}", 
            'Avg Volume': f"{info.get('averageVolume', 'N/A'):,}", 
            'Market Cap': format_dollar(info.get('marketCap')),
            'P/E Ratio': info.get('trailingPE', 'N/A'),
            'Dividend Yield': info.get('dividendYield', 'N/A'),
            'Day Low': format_dollar(info.get('dayLow')),
            'Day High': format_dollar(info.get('dayHigh'))
        }
   
  # Display the table with the stock information 

        st.subheader(f"Summary for {selected_company} ({selected_ticker})")
        # Convert to DataFrame and style the table
        profile_df = pd.DataFrame(profile_summary.items(), columns=["Category", "Value"])  
        with st.container():
            st.write("<style>div.stTable {height: 500px; overflow-y: scroll;}</style>", unsafe_allow_html=True) 
            st.table(profile_df.style.set_table_attributes('class="table"').set_table_styles([{
                'selector': 'th',
                'props': [('background-color', '#D6002A'), ('color', 'white')]
            }]))

    with col2:
        # Time filters
        time_options = {
            '1M': '1mo',
            '3M': '3mo',
            '6M': '6mo',
            'YDT': 'YTD',
            '1Y': '1y',
            '3Y': '3y',
            '5Y': '5y',
            'Max': 'max'
        }
        selected_period = '1Y'  # Default value

        # Buttons to select the period
        time_buttons = st.container()
        cols = time_buttons.columns(len(time_options))
        
        for i, (label, period) in enumerate(time_options.items()):
            if cols[i].button(label):
                selected_period = period

       
        # Get historical data
        try:
            hist_data = get_stock_history(selected_ticker, selected_period)
            if hist_data.empty:
                st.warning(f"No data available for the selected period: {selected_period}")
            else:
               # Create graph with subplots
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # Stock price area plot
                area_plot = go.Scatter(x=hist_data.index, y=hist_data['Close'],
                        fill='tozeroy', fillcolor='rgba(133, 133, 241, 0.2)', showlegend=False)
                fig.add_trace(area_plot, secondary_y=True)


                # Customize the layout
                fig.update_layout(template='plotly_white', height=600)
                fig.update_yaxes(range=[0, hist_data['Close'].max() * 1.1], secondary_y=True)
            

                st.plotly_chart(fig)

        except Exception as e:
            st.error(f"Error getting historical data: {e}")


        
    description = info.get('longBusinessSummary', 'N/A')

    # Create a DataFrame for the description
    description_df = pd.DataFrame({'Item': ['Description'], 'Value': [description]})

    # Show table with description
    st.markdown("<p style='font-size: 16px; color: white; background-color: #FF7993; padding: 10px; border-radius: 5px;'><strong>Company Description</strong></p>", unsafe_allow_html=True)

    # Convert the DataFrame to HTML and hide the header
    description_html = description_df.to_html(index=False, header=False, justify='center')

    # Show table without header
    st.markdown(description_html, unsafe_allow_html=True)

    # Add major shareholder information to the table
    if major_shareholders is not None and not major_shareholders.empty:
        major_shareholders_summary = major_shareholders.head(5).to_string(index=False)
        
        
        st.markdown("<p style='font-size: 16px; color: white; background-color: #FF7993; padding: 10px; border-radius: 5px;'><strong>Major Shareholders</strong></p>", unsafe_allow_html=True)
        st.text(major_shareholders_summary)
    else:
        st.markdown("<p style='font-size: 16px; color: white; background-color: #FF7993; padding: 10px; border-radius: 5px;'><strong>Major Shareholders</strong></p>", unsafe_allow_html=True)
        st.text("Not available")


# **Page 2: Stock Summary**
if page == "Chart":

    
    st.markdown('<h1 style="font-size: 24px; text-decoration: underline;">Financial Chart</h1>', unsafe_allow_html=True)
  
    selected_company = st.selectbox("Select a company", company_names)
    plot_type = st.selectbox("Select Graph type", ["Line plot", "Candlestick"])
    selected_ticker = sp500_table.loc[sp500_table['Security'] == selected_company, 'Symbol'].values[0]
    

    time_options = {
            '1M': '1mo',
            '3M': '3mo',
            '6M': '6mo',
            'YDT': 'YTD',
            '1Y': '1y',
            '3Y': '3y',
            '5Y': '5y',
            'Max': 'max'
        }
    selected_period = '1Y'

    
    time_buttons = st.container()
    cols = time_buttons.columns(len(time_options))
    
    for i, (label, period) in enumerate(time_options.items()):
        if cols[i].button(label):
            selected_period = period
    
    try:
        hist_data = get_stock_history(selected_ticker, selected_period)
        if hist_data.empty:
            st.warning(f"No data available for the selected period: {selected_period}")
        else:
           
            hist_data['SMA_50'] = hist_data['Close'].rolling(window=50).mean()
            fig = make_subplots(specs=[[{'secondary_y': True}]])

          
            if plot_type=='Line plot':

                price_trace = go.Scatter(
                    x=hist_data.index,
                    y=hist_data['Close'],
                    mode='lines',
                    name='Closing Price',
                    line=dict(color='blue'),
                    fill='tozeroy',
                    fillcolor='rgba(0, 100, 250, 0.3)'  
                )
                fig.add_trace(price_trace, secondary_y=True)

                # Add SMA trace
                sma_trace = go.Scatter(
                x=hist_data.index,
                y=hist_data['SMA_50'],
                mode='lines',
                name='SMA 50 days',
                line=dict(color='orange', dash='dash')  # Dashed orange line for SMA
                )
                fig.add_trace(sma_trace, secondary_y=True)


            elif plot_type == "Candlestick":
                # Candlestick plot
                candle_trace = go.Candlestick(
                    x=hist_data.index,
                    open=hist_data['Open'],
                    high=hist_data['High'],
                    low=hist_data['Low'],
                    close=hist_data['Close'],
                    name='Candlestick',
                    increasing_line_color='green', 
                    decreasing_line_color='red'  
                    )
                fig.add_trace(candle_trace, secondary_y=True)
                sma_trace = go.Scatter(
                x=hist_data.index,
                y=hist_data['SMA_50'],
                mode='lines',
                name='SMA 50 days',
                line=dict(color='orange', dash='dash')  # Dashed orange line for SMA
                )
                fig.add_trace(sma_trace, secondary_y=True)
                fig.update_layout(xaxis_rangeslider_visible=False)

          # Volume chart
            volume_trace = go.Bar(
                x=hist_data.index,
                y=hist_data['Volume'],
                name='Volume',
                marker_color=np.where(hist_data['Close'].pct_change() > 0, 'green', 'red')
            )
            fig.add_trace(volume_trace, secondary_y=False)

          
            fig.update_layout(
                title=f'Chart of {selected_company} ({selected_ticker}) for the period:{selected_period}',
                yaxis=dict(title='Volumen'),
                yaxis2=dict(title='Price (USD)', overlaying='y', side='right'),
                xaxis=dict(title='Date'),
                barmode='overlay',
                template='plotly_dark')

            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error getting historical data: {e}")




# **Page 3: Stock Summary**
if page == "Financials":


    st.markdown('<h1 style="font-size: 24px; text-decoration: underline;">Financials</h1>', unsafe_allow_html=True)


    selected_company = st.selectbox("Select a company", company_names)
    selected_ticker = sp500_table.loc[sp500_table['Security'] == selected_company, 'Symbol'].values[0]
    stock = yf.Ticker(selected_ticker)
    
     

# Collect and present the financial information of the stock
    
        # Retrieve financial information
    financials = stock.financials
    info = stock.info

            # Display basic financial metrics
        
    st.write(f"**Company Name:** {info.get('longName', 'N/A')}")
    st.write(f"**Ticker:** {selected_ticker}")
    st.write(f"**Market Capitalization:** ${info.get('marketCap', 'N/A') / 1e9:.2f}B")
    st.write(f"**P/E Ratio:** {info.get('forwardPE', 'N/A')}")
    st.write(f"**EPS:** ${info.get('trailingEps', 'N/A')}")
    st.write(f"**Price/Sales Ratio:** {info.get('priceToSalesTrailing12Months', 'N/A')}")
    st.write(f"**Revenue:** ${info.get('totalRevenue', 'N/A') / 1e6:.2f}M")
        

    period_options = st.radio("Select a period:", ["Annual", "Quarterly"])

    statement_options = ["Income Statement", "Balance Sheet", "Cash Flow"]
    st.markdown("### Financial Statement")
    cols = st.columns(len(statement_options)) 

    
    selected_statement = None
    
    
    for i, option in enumerate(statement_options):
        if cols[i].button(option):
            selected_statement = option  

   
    if selected_statement:
        st.subheader(selected_statement)
    
    

    try:

        def process_financial_data(df):
        
            df.columns = pd.to_datetime(df.columns).year
       
            df = df[sorted(df.columns)]
            df = df.applymap(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            return df

        def process_financial_data(df, is_quarterly=False):
            
            df.columns = pd.to_datetime(df.columns, errors='coerce')

            if is_quarterly:
              
                df.columns = df.columns.to_series().apply(lambda x: f"{x.year} Q{x.quarter}")
            else:
               
                df.columns = df.columns.year
           
            
            df = df[sorted(df.columns)]
            
            
            df = df.applymap(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            return df





        
        styles = [
            {'selector': 'thead th', 'props': [('background-color', '#D6002A'), ('color', 'white')]},
            {'selector': 'thead tr:nth-child(1) th', 'props': [('background-color', '#D6002A'), ('color', 'white')]}
        ]

        if selected_statement:
            st.subheader(f"{selected_statement} ({period_options})")
            is_quarterly = period_options == "Quarterly"  


            if selected_statement == "Income Statement":
                    income_statement = stock.financials if period_options == "Annual" else stock.quarterly_financials
                    if not income_statement.empty:
                        processed_income = process_financial_data(income_statement, is_quarterly)
                        st.table(processed_income.style.set_table_styles(styles))
                    else:
                        st.warning("No data available for the income statement.")

            elif selected_statement == "Balance Sheet":
                    balance_sheet = stock.balance_sheet if period_options == "Annual" else stock.quarterly_balance_sheet
                    if not balance_sheet.empty:
                        processed_balance = process_financial_data(balance_sheet , is_quarterly)
                        st.table(processed_balance.style.set_table_styles(styles))
                    else:
                        st.warning("No data available for the balance sheet.")

            elif selected_statement == "Cash Flow":
                    cash_flow = stock.cashflow if period_options == "Annual" else stock.quarterly_cashflow
                    if not cash_flow.empty:
                        processed_cashflow = process_financial_data(cash_flow, is_quarterly)
                        st.table(processed_cashflow.style.set_table_styles(styles))
                    else:
                        st.warning("No data available for cash flow.")

    except Exception as e:
            st.error(f"Error retrieving financial statement: {e}")

# **Page 4: Stock Summary**
if page == "Monte Carlo simulation":

    
    st.markdown('<h1 style="font-size: 24px; text-decoration: underline;">Monte Carlo simulation</h1>', unsafe_allow_html=True)

   
 # Simulation parameters
    Sim = {
            '200': '200',
            '500': '500',
            '1000': '1000'
        }
    

    TimeHor = {
            '30': '30',
            '60': '60',
            '90': '90'
        }

    col1, col2, col3 , col4= st.columns(4) 


    with col1:
        selected_company = st.selectbox("Select a Company", company_names)
   
        selected_ticker = sp500_table.loc[sp500_table['Security'] == selected_company, 'Symbol'].values[0]
    
        selected_period = '1Y' 

    with col2:
        time_horizon = int(st.selectbox("Time horizon", TimeHor))
    with col3:
        n_simulation = int(st.selectbox("Number of simulations", Sim))
    with col4:
        seed = st.number_input("Seed", value=123)


    hist_data = get_stock_history(selected_ticker, selected_period)

        # Defind the run_simulation() function
    @st.cache_data
    def run_simulation(stock_price, time_horizon, n_simulation, seed):
        """
        This function run the Monte Carlo simulation using the input parameters.
        
        Input:
            - stock_price : A DataFrame store the stock price (from Yahoo Finance)
            - time_horizon : Period of the simulation (in days)
            - n_simulation : Number of simulations
            - seed : The ramdom seed
            
        Output:
            - simulation_df : A DataFrame stores the simulated prices
        """
        
        # Calculate some financial metrics for the simulation
        # Daily return (of close price)
        daily_return = stock_price['Close'].pct_change()
        # Daily volatility (of close price)
        daily_volatility = np.std(daily_return)

        # Run the simulation
        np.random.seed(seed)
        simulation_df = pd.DataFrame()  # Initiate the data frame

        for i in range(n_simulation):

            # The list to store the next stock price
            next_price = []

            # Create the next stock price
            last_price = stock_price['Close'].iloc[-1]

            for j in range(time_horizon):

                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price

            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            simulation_df = pd.concat([simulation_df, next_price_df], axis=1)

        return simulation_df

    # Test the function
    simulation_df = run_simulation(stock_price=hist_data,
                                    time_horizon=time_horizon,
                                    n_simulation=n_simulation,
                                    seed=seed)
    

    def plot_simulation_price(stock_price, simulation_df):
        """
        This function plot the simulated stock prices using line plot.
        
        Input:
            - stock_price : A DataFrame store the stock price (from Yahoo Finance)
            - simulation_df : A DataFrame stores the simulated prices
            
        Output:
            - Plot the stock prices
        """
            
        # Plot the simulation stock price in the future
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(simulation_df)
        ax.set_title('Monte Carlo simulation for the stock price in next ' + str(simulation_df.shape[0]) + ' days')
        ax.set_xlabel('Day')
        ax.set_ylabel('Price')
        ax.axhline(y=stock_price['Close'].iloc[-1], color='red')
        ax.legend(['Current stock price is: ' + str(np.round(stock_price['Close'].iloc[-1], 2))])
        ax.get_legend().legend_handles[0].set_color('red')
    
        st.pyplot(fig)

    # Test the function
    plot_simulation_price(hist_data, simulation_df)


        # Defind the value_at_risk() function:
    def value_at_risk(stock_price, simulation_df):
        """
        This function calculate the Value at Risk (VaR) of the stock based on the Monte Carlo simulation.
        
        Input:
            - stock_price : A DataFrame store the stock price (from Yahoo Finance)
            - simulation_df : A DataFrame stores the simulated prices
            
        Output:
            - VaR value
        """
            
        # Price at 95% confidence interval
        future_price_95ci = np.percentile(simulation_df.iloc[-1:, :].values[0, ], 5)

        # Value at Risk
        VaR = stock_price['Close'].iloc[-1] - future_price_95ci
        st.write(f"VaR at 95% confidence interval is:${VaR:.2f} USD")
        
        
    # Test the function
    value_at_risk(hist_data, simulation_df)

if page == 'Relative Performance Analysis':

    st.markdown("""
        <h2 style="text-decoration: underline;">Relative Performance Analysis</h2>
        <p style="font-size: 16px;">
        This analysis compares the performance (or return) of individual stocks against the S&P 500 index over a selected period. <br><br>
        - Return measures the percentage change in the price of a stock over a specific time period. It indicates how much an investment has gained or lost in value.
        - In this context, we use the relative performance to see whether a stock has outperformed (done better) or underperformed (done worse) compared to the S&P 500 index.
        </p>
        """, unsafe_allow_html=True)
    

    col1, col2= st.columns([1, 2])  
    with col1:
        selected_company = st.selectbox("Select a company", company_names)
        selected_ticker = sp500_table.loc[sp500_table['Security'] == selected_company, 'Symbol'].values[0]
    with col2:
        time_options = {
                '1M': '1mo',
                '3M': '3mo',
                '6M': '6mo',
                'YDT': 'YTD',
                '1Y': '1y',
                '3Y': '3y',
                '5Y': '5y',
                'Max': 'max'
            }
        selected_period = '1Y'

        
        time_buttons = st.container()
        cols = time_buttons.columns(len(time_options))
        
        for i, (label, period) in enumerate(time_options.items()):
            if cols[i].button(label):
                selected_period = period


    @st.cache_data
    def get_adjusted_close(ticker, selected_period):
        stock = yf.Ticker(ticker)
        hist = stock.history(period=selected_period)
        return hist['Close']

    stock_data = get_adjusted_close(selected_ticker, selected_period)
    sp500_data = get_adjusted_close("^GSPC", selected_period)

    stock_returns = (stock_data / stock_data.iloc[0]) * 100 - 100
    sp500_returns = (sp500_data / sp500_data.iloc[0]) * 100 - 100

    
    fig = go.Figure()

    
    fig.add_trace(go.Scatter(
        x=stock_returns.index, 
        y=stock_returns, 
        mode='lines', 
        name=selected_company,
        line=dict(color='blue')
    ))

    
    fig.add_trace(go.Scatter(
        x=sp500_returns.index, 
        y=sp500_returns, 
        mode='lines', 
        name="S&P 500",
        line=dict(color='red', dash='dash')
    ))

    
    fig.update_layout(
        title=f"Relative Performance of {selected_company} vs S&P 500 in the period {selected_period}",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Performance (%)"),
        template="plotly_dark",
    )

    st.plotly_chart(fig)




#==============================================================================

#Individual assignment for Angie Tatiana Vargas Perea - MBD 2024

# Code adapted from examples provided by Professor PHAN Minh. 

