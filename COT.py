import os
import pandas as pd
import requests, zipfile, io
from datetime import date
from bs4 import BeautifulSoup

# cot_hist - downloads compressed bulk files
def cot_hist(cot_report_type="legacy_futopt", store_txt=True, verbose=True):
    '''Downloads the compressed COT report historical data of the selected report type
    starting from, depending on the selected report type, 1986, 1995 or 2006 until 2016
    from the cftc.gov webpage as zip file, unzips the downloaded folder and returns
    the cot data as DataFrame.  
    
    COT report types:  
    "legacy_fut" as report type argument selects the Legacy futures only report,
    "legacy_futopt" the Legacy futures and options report,
    "supplemental_futopt" the Sumpplemental futures and options reports,
    "disaggregated_fut" the Disaggregated futures only report, 
    "disaggregated_futopt" the COT Disaggregated futures and options report, 
    "traders_in_financial_futures_fut" the Traders in Financial Futures futures only report, and 
    "traders_in_financial_futures_fut" the Traders in Financial Futures futures and options report. 
    
    Args:
        cot_report_type (str): selection of the COT report type. Defaults to "legacy_fut" (Legacy futures only report). 
    
    Returns:
        A DataFrame with differing variables (depending on the selected report type). 
        
    Raises:
        ValueError: Raises an exception and returns the argument options.'''    
    try: 
        if cot_report_type == "legacy_fut": 
            url_end = "deacot1986_2016"
            txt = "FUT86_16.txt"
            if verbose: print("Selected: COT Legacy report. Futures only.")
        elif cot_report_type == "legacy_futopt": 
            url_end = "deahistfo_1995_2016"
            txt = "Com95_16.txt"
            if verbose: print("Selected: COT Legacy report. Futures and Options.")
        elif cot_report_type == "supplemental_futopt": 
            url_end = "dea_cit_txt_2006_2016"
            txt = "CIT06_16.txt"
            if verbose: print("Selected: COT Supplemental report. Futures and Options.")
        elif cot_report_type == "disaggregated_fut": 
            url_end = "fut_disagg_txt_hist_2006_2016"
            txt = "F_Disagg06_16.txt"
            if verbose: print("Selected: COT Disaggregated report. Futures only.")
        elif cot_report_type == "disaggregated_futopt": 
            url_end = "com_disagg_txt_hist_2006_2016"
            txt = "C_Disagg06_16.txt"
            if verbose: print("Selected: COT Disaggregated report. Futures and Options.")
        elif cot_report_type == "traders_in_financial_futures_fut": 
            url_end = "fin_fut_txt_2006_2016"
            txt = "F_TFF_2006_2016.txt" 
            if verbose: print("Selected: COT Traders in Financial Futures report. Futures only.")
        elif cot_report_type == "traders_in_financial_futures_futopt": 
            url_end = "fin_com_txt_2006_2016"
            txt = "C_TFF_2006_2016.txt" 
            if verbose: print("Selected: COT Traders in Financial Futures report. Futures and Options.")
    except ValueError:    
        if verbose: print("""Input needs to be either:
                "legacy_fut", "legacy_futopt", "supplemental_futopt",
                "disaggregated_fut", "disaggregated_futopt", 
                "traders_in_financial_futures_fut" or
                "traders_in_financial_futures_futopt" """)
    
    cot_url = "https://cftc.gov/files/dea/history/" + str(url_end) + ".zip"
    req = requests.get(cot_url) 
    z = zipfile.ZipFile(io.BytesIO(req.content))
    z.extractall()
    df = pd.read_csv(txt, low_memory=False)
    if store_txt:
        if verbose: print("Stored the extracted file", txt, "in the working directory.")
    else:
        os.remove(txt)
    return df

# cot_year - downloads single years
def cot_year(year=2020, cot_report_type="legacy_fut", store_txt=True, verbose=True):    
    '''Downloads the selected COT report historical data for a single year
    from the cftc.gov webpage as zip file, unzips the downloaded folder and returns
    the cot data as DataFrame.
    For the current year selection, please note: updates by the CFTC occur typically weekly.
    Once the documents update by CFTC occured, the updated data can be accessed through 
    this function. The cot_report_type must match one of the following.

    COT report types:  
    "legacy_fut" as report type argument selects the Legacy futures only report,
    "legacy_futopt" the Legacy futures and options report,
    "supplemental_futopt" the Supplemental futures and options reports,
    "disaggregated_fut" the Disaggregated futures only report, 
    "disaggregated_futopt" the COT Disaggregated futures and options report, 
    "traders_in_financial_futures_fut" the Traders in Financial Futures futures only report, and 
    "traders_in_financial_futures_futopt" the Traders in Financial Futures futures and options report. 
    
    Args:
        cot_report_type (str): selection of the COT report type. Defaults to "legacy_fut" (Legacy futures only report).
        year (int): year specification as YYYY
    
    Returns:
        A DataFrame with differing variables (depending on the selected report type). 
        
    Raises:
        ValueError: Raises an exception and returns the argument options.'''    
    if verbose: print("Selected:", cot_report_type)
    try: 
        if cot_report_type == "legacy_fut": 
            rep = "deacot"
            txt = "annual.txt"
        elif cot_report_type == "legacy_futopt": 
            rep = "deahistfo"
            txt = "annualof.txt"
        elif cot_report_type == "supplemental_futopt": 
            rep = "dea_cit_txt_"
            txt = "annualci.txt"
        elif cot_report_type == "disaggregated_fut": 
            rep = "fut_disagg_txt_"
            txt = "f_year.txt"
        elif cot_report_type == "disaggregated_futopt": 
            rep = "com_disagg_txt_"
            txt = "c_year.txt"
        elif cot_report_type == "traders_in_financial_futures_fut": 
            rep = "fut_fin_txt_"
            txt = "FinFutYY.txt"
        elif cot_report_type == "traders_in_financial_futures_futopt": 
            rep = "com_fin_txt_"
            txt = "FinComYY.txt"
    except ValueError:    
        print("""Input needs to be either:
                "legacy_fut", "legacy_futopt", "supplemental_futopt",
                "disaggregated_fut", "disaggregated_futopt", 
                "traders_in_financial_futures_fut" or
                "traders_in_financial_futures_futopt" """)
    
    cot_url = "https://cftc.gov/files/dea/history/" + rep + str(year) + ".zip"
    r = requests.get(cot_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    df = pd.read_csv(txt, low_memory=False)  
    if verbose: print("Downloaded single year data from:", year)
    if store_txt:
        if verbose: print("Stored the file", txt, "in the working directory.")
    else:
        os.remove(txt)
    return df

import os
import pandas as pd

# Directory containing the COT history CSV files
data_directory = '.venv/Petroleum/commodities_ICE/ICE_COTHist'

# List of CSV files from 2013 to 2024
file_list = [f'COTHist{year}.csv' for year in range(2013, 2025)]

# Load and combine all the CSV files into a single DataFrame
combined_df_list = []

for file_name in file_list:
    file_path = os.path.join(data_directory, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        combined_df_list.append(df)
    else:
        print(f"File not found: {file_path}")

combined_df = pd.concat(combined_df_list, ignore_index=True)


# Filter for the specific products of interest
products = ['ICE Brent Crude Futures and Options - ICE Futures Europe', 
            'ICE Dubai 1st Line Futures', 'GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE', 
            'NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE', 
            'ICE Gasoil Futures and Options - ICE Futures Europe', 
            'WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE', 
            'CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE']
'''
ADDITION
'''
print(df)
df_copy = combined_df.copy()
names = df_copy['Market_and_Exchange_Names']

names = names.drop_duplicates()
# Filtering names that contain 'GASOIL'
filtered_names = names[names.str.contains('Gasoil', case=False)]

# Printing the filtered names
print('Gasoil Name')
for name in filtered_names:
    print(name)
    
'''
END ADDITION
'''

combined_df = combined_df[combined_df['Market_and_Exchange_Names'].isin(products)]





def collect_years(start_year, end_year, cot_report_type, directory):
    os.makedirs(directory, exist_ok=True)
    all_data = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(directory, f"COT_{cot_report_type}_{year}.csv")
        
        if os.path.exists(file_path):
            # If file exists, load it
            print(f"Loading data from {file_path}")
            df = pd.read_csv(file_path)
        else:
            # If file doesn't exist, download and store it
            print(f"Downloading data for year {year}")
            df = cot_year(year=year, cot_report_type=cot_report_type, store_txt=False, verbose=True)
            df.to_csv(file_path, index=False)
        
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
   
    return combined_df

# Directory to store the files
directory = ".venv/Petroleum/commodities_ICE/CFTC_COTHist"

# Collect data from 2013 to 2024
df = collect_years(2013, 2024, cot_report_type="disaggregated_futopt", directory=directory)



df = df[df['Market_and_Exchange_Names'].isin(products)]
# Sort DataFrame by date
combined_df = combined_df.sort_values(by='As_of_Date_In_Form_YYMMDD')

# Define columns to calculate changes for
columns_to_change = ['M_Money_Positions_Short_All','M_Money_Positions_Long_All','Prod_Merc_Positions_Long_All','Prod_Merc_Positions_Short_All',
                     'Swap_Positions_Long_All','Swap_Positions_Short_All','Other_Rept_Positions_Short_All','Other_Rept_Positions_Long_All',
                     'NonRept_Positions_Long_All','NonRept_Positions_Short_All']

# Group by 'Market_and_Exchange_Names' and calculate changes
grouped = combined_df.groupby('Market_and_Exchange_Names')

# Function to calculate changes and combine the groups
def calculate_changes(grouped):
    grouped_dfs = []
    for group_name, group_data in grouped:
        for column in columns_to_change:
            change_column = 'Change_in_' + column.replace('_Positions_', '_')
            group_data[change_column] = group_data[column].diff().fillna(0)
        grouped_dfs.append(group_data)
    return pd.concat(grouped_dfs)

combined_df = calculate_changes(grouped)

# Combine and filter data
combined_df = pd.concat([df, combined_df ], ignore_index=True)
combined_df = combined_df[combined_df['Market_and_Exchange_Names'].isin(products)]

# Process columns and data for net positions
market_participants = ['Prod_Merc', 'Swap', 'M_Money', 'Other_Rept', 'Tot_Rept', 'NonRept']

def create_net_columns(df, participants):
    for participant in participants:
        long_col = f'{participant}_Positions_Long_All'
        short_col = f'{participant}_Positions_Short_All'
        net_col = f'{participant}_Positions_Net_All'
        
        change_long_col = f'Change_in_{participant}_Long_All'
        change_short_col = f'Change_in_{participant}_Short_All'
        change_net_col = f'Change_in_{participant}_Net_All'
        
        df[[long_col, short_col, change_long_col, change_short_col]] = df[
            [long_col, short_col, change_long_col, change_short_col]
        ].apply(pd.to_numeric, errors='coerce')
        
        df[net_col] = df[long_col] - df[short_col]
        df[change_net_col] = df[net_col] - ((df[long_col] - df[change_long_col]) - (df[short_col] - df[change_short_col]))
    
    return df

combined_df = create_net_columns(combined_df, market_participants)

# Melt and reshape the DataFrame
value_vars = []
for participant in market_participants:
    value_vars.extend([
        f'{participant}_Positions_Long_All',
        f'{participant}_Positions_Short_All',
        f'{participant}_Positions_Net_All',
        f'Change_in_{participant}_Long_All',
        f'Change_in_{participant}_Short_All',
        f'Change_in_{participant}_Net_All'
    ])

df_melted = pd.melt(combined_df, id_vars=['Market_and_Exchange_Names', 'As_of_Date_In_Form_YYMMDD'], 
                    value_vars=value_vars,
                    var_name='Type', value_name='Value')

df_melted['Position'] = df_melted['Type'].apply(lambda x: 'Long' if 'Long' in x else ('Short' if 'Short' in x else 'Net'))
df_melted['Metric'] = df_melted['Type'].apply(lambda x: 'Value' if 'Change' not in x else 'Change')
df_melted['Participant'] = df_melted['Type'].apply(lambda x: next(participant for participant in market_participants if participant in x))

df_melted.drop('Type', axis=1, inplace=True)

# Create the final pivot table
def create_final_pivot_table(df, index, columns, values, aggfunc):
    pivot_table = pd.pivot_table(
        df,
        index=index,
        columns=columns,
        values=values,
        aggfunc=aggfunc
    )
    pivot_table.columns = ['_'.join(col).strip() for col in pivot_table.columns.values]
    return pivot_table

final_pivot_tables = {}
unique_markets = df_melted['Market_and_Exchange_Names'].unique()

for market in unique_markets:
    df_subset = df_melted[df_melted['Market_and_Exchange_Names'] == market]
    pivot_table = create_final_pivot_table(df_subset, index='As_of_Date_In_Form_YYMMDD', columns=['Participant', 'Position', 'Metric'], values='Value', aggfunc='sum')
    final_pivot_tables[market] = pivot_table

# Example to access the final pivot tables for each market
for market in unique_markets:
    print(f"Pivot Table for {market}:")
    print(final_pivot_tables[market])
    print("\n")


# Iterate over the dictionary and convert the index for each dataframe
for key, df in final_pivot_tables.items():
    # Reset the index to access the 'As_of_Date_In_Form_YYMMDD' column
    df.reset_index(inplace=True)
    
    # Convert the 'As_of_Date_In_Form_YYMMDD' column to a datetime column
    df['Date'] = pd.to_datetime(df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
    
    # Optionally, if you want to replace the original column with the datetime column
    # df['As_of_Date_In_Form_YYMMDD'] = pd.to_datetime(df['As_of_Date_In_Form_YYMMDD'], format='%y%m%d')
    
    # Set the 'Date' column back as the index
    df.set_index('Date', inplace=True)
    
    # Optionally, drop the original 'As_of_Date_In_Form_YYMMDD' column if you no longer need it
    # df.drop(columns=['As_of_Date_In_Form_YYMMDD'], inplace=True)
    
    # Update the dataframe in the dictionary
    final_pivot_tables[key] = df
    
 # Create the combined pivot tables as requested
# Retrieve the pivot tables for "WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE" and "CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE"
wti_physical_nymex = final_pivot_tables["WTI-PHYSICAL - NEW YORK MERCANTILE EXCHANGE"]
crude_oil_light_sweet_wti_ice = final_pivot_tables["CRUDE OIL, LIGHT SWEET-WTI - ICE FUTURES EUROPE"]

brent_ice = final_pivot_tables["ICE Brent Crude Futures and Options - ICE Futures Europe"]
european_gasoil = final_pivot_tables["ICE Gasoil Futures and Options - ICE Futures Europe"]

european_gasoil.loc[:, european_gasoil.columns != 'As_of_Date_In_Form_YYMMDD'] *= 745/1000

print(european_gasoil)
final_pivot_tables["ICE Gasoil Futures and Options - ICE Futures Europe"] = european_gasoil

us_diesel = final_pivot_tables["NY HARBOR ULSD - NEW YORK MERCANTILE EXCHANGE"]

RBOB = final_pivot_tables["GASOLINE RBOB - NEW YORK MERCANTILE EXCHANGE"]

# Align the two pivot tables on the index to ensure they are compatible for summation
# Find the intersection of the indices (dates) for matching dates only
matching_dates_wti = wti_physical_nymex.index.intersection(crude_oil_light_sweet_wti_ice.index)
wti_physical_nymex = wti_physical_nymex.loc[matching_dates_wti]
crude_oil_light_sweet_wti_ice = crude_oil_light_sweet_wti_ice.loc[matching_dates_wti]

# Sum the corresponding columns
wti_combined = wti_physical_nymex.add(crude_oil_light_sweet_wti_ice, fill_value=0)

# Print the new pivot table
print("Pivot Table for WTI (NYMEX + ICE):")
print(wti_combined)
final_pivot_tables["WTI (NYMEX + ICE)"] = wti_combined
# Find the intersection of the indices (dates) for matching dates only for crude oil combination
matching_dates_crude_oil = brent_ice.index.intersection(wti_combined.index)
brent_ice = brent_ice.loc[matching_dates_crude_oil]
wti_combined = wti_combined.loc[matching_dates_crude_oil]

crude_oil = brent_ice.add(wti_combined, fill_value=0)

print('Crude Oil:')
print(crude_oil)
final_pivot_tables["CRUDE OIL (WTI NYMEX + WTI ICE + BRENT)"] = crude_oil
# Find the intersection of the indices (dates) for matching dates only for middle distillates
matching_dates_middle_distillates = european_gasoil.index.intersection(us_diesel.index)
european_gasoil = european_gasoil.loc[matching_dates_middle_distillates]



us_diesel = us_diesel.loc[matching_dates_middle_distillates]

middle_distillates = european_gasoil.add(us_diesel, fill_value=0)

matching_dates_products = RBOB.index.intersection(middle_distillates.index)

RBOB = RBOB.loc[matching_dates_products]

products = middle_distillates.add(RBOB, fill_value=0)

everything_matching_dates = products.index.intersection(crude_oil.index)

petroleum = crude_oil.add(products, fill_value=0)


print('Middle Distillates:')
print(middle_distillates)

final_pivot_tables["MIDDLE DISTILLATES (U.S Diesel + EU Gasoil)"] = middle_distillates

final_pivot_tables["PRODUCTS (U.S Diesel + EU Gasoil + RBOB)"] = products

final_pivot_tables["Total Petroleum"] = petroleum

print(petroleum.columns)
 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd

# Assuming final_pivot_tables is a dictionary with pivot tables for each market
# Example: final_pivot_tables is a dictionary with pivot tables for each market

# Function to filter data based on the number of years
def get_lastdata(df, nb_of_years):
    df['Date'] = pd.to_datetime(df.index)
    months = round(nb_of_years * 12)
    last_years_date = pd.Timestamp.today() - pd.DateOffset(months=months)
    df_last_years = df[df['Date'] >= last_years_date]
    df_last_years.reset_index(drop=True, inplace=True)
    return df_last_years

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Layout of the dashboard
app.layout = html.Div([
    html.H1("Futures Market Comparison and Analysis"),
    html.Div([
        dcc.Dropdown(
            id='market-dropdown',
            options=[{'label': market, 'value': market} for market in final_pivot_tables.keys()],
            value=list(final_pivot_tables.keys())[0],  # Default value
            style={'width': '50%', 'display': 'inline-block'}
        ),
        dcc.Input(
            id='years-input',
            type='number',
            value=5,
            min=0,
            step=0.4,
            placeholder="Years Back",
            style={'width': '20%', 'display': 'inline-block', 'marginLeft': '20px'}
        )
    ]),
    dcc.Graph(id='sentiment-chart', style={'height': '700px'}),
    dcc.Graph(id='net-positions-chart', style={'height': '700px'}),
    dcc.Graph(id='positions-change-chart'),
    dcc.Graph(id='net-long-change-chart'),  # Add the new net-long change chart
    dcc.Graph(id='change-net-change-chart')
])

# Callback to update the charts based on the selected market and years
@app.callback(
    [Output('sentiment-chart', 'figure'),
     Output('net-positions-chart', 'figure'),
     Output('positions-change-chart', 'figure'),
     Output('net-long-change-chart', 'figure'),
     Output('change-net-change-chart', 'figure')],
    [Input('market-dropdown', 'value'),
     Input('years-input', 'value')]
)
def update_charts(selected_market, years_back):
    df = final_pivot_tables[selected_market]
    df = get_lastdata(df, years_back)
    
    sentiment_chart = create_sentiment_chart(df, selected_market)
    net_positions_chart = create_net_positions_chart(df, selected_market)
    positions_change_chart = create_positions_change_chart(df, selected_market)
    net_long_change_chart = create_change_net_long_chart(df, selected_market)
    change_net_change_chart = create_change_net_change_chart(df, selected_market)
    
    return sentiment_chart, net_positions_chart, positions_change_chart,  net_long_change_chart, change_net_change_chart

def create_sentiment_chart(df, market):
    fig = go.Figure()

    # Money Manager Long Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['M_Money_Long_Value'], 
        mode='lines', 
        name='Money Manager Long',
        line=dict(color='green'),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 0, 0.2)',
    ))

    # Money Manager Short Position as Negative
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=-df['M_Money_Short_Value'], 
        mode='lines', 
        name='Money Manager Short',
        line=dict(color='red'),
        fill='tozeroy', 
        fillcolor='rgba(255, 0, 0, 0.2)',
    ))

    # Money Manager Net Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['M_Money_Net_Value'], 
        mode='lines', 
        name='Money Manager Net',
        line=dict(color='blue'),
    ))

    # Ratio of Long to Short Positions
    ratio = df['M_Money_Long_Value'] / df['M_Money_Short_Value'].replace({0: 1})  # Avoid division by zero
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=ratio, 
        mode='lines', 
        name='Long/Short Ratio',
        line=dict(color='orange', dash='dash'),
        yaxis='y2',  # Associate this trace with the secondary y-axis
    ))

    fig.update_layout(
        title=f'Sentiment for {market}', 
        xaxis_title='Date', 
        yaxis_title='Contracts',
        yaxis2=dict(
            title="Long/Short Ratio",
            overlaying='y',
            side='right',  # Ensure the secondary y-axis is on the right side
            showgrid=False,
            zeroline=False,
        ),
        hovermode='x unified',
        template='plotly_dark'
    )

    return fig

# Function to create the net positions chart
def create_net_positions_chart(df, market):
    fig = go.Figure()

    # Producer/Merchant Net Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Prod_Merc_Net_Value'], 
        mode='lines', 
        name='Producer/Merchant Net',
        line=dict(color='blue'),
        fill='tozeroy', 
        fillcolor='rgba(0, 0, 255, 0.2)'
    ))

    # Swap Dealer Net Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Swap_Net_Value'], 
        mode='lines', 
        name='Swap Dealer Net',
        line=dict(color='purple'),
        fill='tozeroy', 
        fillcolor='rgba(128, 0, 128, 0.2)'
    ))

    # Money Manager Net Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['M_Money_Net_Value'], 
        mode='lines', 
        name='Money Manager Net',
        line=dict(color='green'),
        fill='tozeroy', 
        fillcolor='rgba(0, 255, 0, 0.2)'
    ))

    # Other Reportable Net Position
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Other_Rept_Net_Value'], 
        mode='lines', 
        name='Other Reportable Net',
        line=dict(color='orange'),
        fill='tozeroy', 
        fillcolor='rgba(255, 165, 0, 0.2)'
    ))

    fig.update_layout(
        title=f'Net Positions for {market}', 
        xaxis_title='Date', 
        yaxis_title='Net Positions',
        hovermode='x unified',
        template='plotly_dark'
    )
    return fig

# Function to create the positions change chart
def create_positions_change_chart(df, market):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['M_Money_Long_Change'], name='Money Manager Long Change', marker_color='green'))
    fig.add_trace(go.Bar(x=df['Date'], y=df['M_Money_Short_Change'], name='Money Manager Short Change', marker_color='red'))
    fig.update_layout(title=f'Positions Change for {market}', xaxis_title='Date', yaxis_title='Change in Positions', template='plotly_dark')
    return fig

def create_change_net_long_chart(df, market):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df['Date'], y=df['M_Money_Net_Change'], name='Money Manager Net-Long Change', marker_color='blue'))
    fig.update_layout(title=f'Positions Change for {market}', xaxis_title='Date', yaxis_title='Change in Positions', template='plotly_dark')
    return fig

# Function to create the change and net change chart for the last date
def create_change_net_change_chart(df, market):
    last_row = df.iloc[-1]
    categories = ['Prod_Merc', 'Swap', 'M_Money', 'Other_Rept']
    
    data = []
    for cat in categories:
        data.append({
            'category': cat,
            'long_change': last_row[f'{cat}_Long_Change'],
            'short_change': last_row[f'{cat}_Short_Change'],
            'net_change': last_row[f'{cat}_Net_Change']
        })

    fig = go.Figure()

    for d in data:
        fig.add_trace(go.Bar(
            x=[d['category']], 
            y=[d['long_change']], 
            name=f"{d['category']} Long Change",
            marker_color='green'
        ))
        fig.add_trace(go.Bar(
            x=[d['category']], 
            y=[-d['short_change']], 
            name=f"{d['category']} Short Change",
            marker_color='red'
        ))
        fig.add_trace(go.Bar(
            x=[d['category']], 
            y=[d['net_change']], 
            name=f"{d['category']} Net Change",
            marker_color='blue'
        ))

    fig.update_layout(
        title=f'Change and Net Change for {market} (Last Date)', 
        xaxis_title='Category', 
        yaxis_title='Change in Positions',
        barmode='group',
        template='plotly_dark'
    )
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
