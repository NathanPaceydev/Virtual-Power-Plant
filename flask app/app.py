from flask import Flask, render_template, request, redirect, url_for, session, Response
import requests
import openmeteo_requests
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
import calendar
import requests_cache
from retry_requests import retry
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"
import csv

import plotly.graph_objs as go
from plotly.graph_objs import Scatter, Figure

from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import glob
import os



app = Flask(__name__)

app.secret_key = 'your_secret_key_here'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Reset the necessary session variables to zero
        session_keys = ['surface_area', 'postal_code', 'array_type', 'module_type', 'tilt',
                        'num_turbines', 'turbineHeight', 'battery_consumption',
                        'battery_runtime', 'battery_final_percent', 'battery_max_cycles']
        for key in session_keys:
            session[key] = 0  # Resetting each to zero
            
        for key in [ 'wind_visited', 'battery_visited']:
            session.pop(key, None)
            
        # Store existing form data in session
        session['surface_area'] = request.form.get('surfaceArea')
        if session['surface_area'] == '':
            session['surface_area'] = 0
            
        session['postal_code'] = request.form.get('postalCode')
        session['array_type'] = request.form.get('arrayType')
        session['module_type'] = request.form.get('moduleType')
        session['tilt'] = request.form.get('tilt')

        # Store the new input for the number of wind turbines
        session['num_turbines'] = request.form.get('numTurbines', type=int) or 0
        session['turbineHeight'] = request.form.get('turbineHeight', type=int)
        
        # battery consumption
        session['battery_consumption'] = request.form.get('hourlyDemand', type=int)
        session['battery_runtime'] = request.form.get('runtime', type=int)
        session['battery_final_percent'] = request.form.get('finalPercentage', type=int)
        session['battery_max_cycles'] = request.form.get('maxCycles', type=int)
        
        # Redirect to the solar page
        return redirect(url_for('solar'))

    return render_template('home.html')

@app.route('/location', methods=['GET','POST'])
def location():
    postal_code = session.get('postal_code', 'Not provided')
    latitude = session.get('latitude', 'Not provided')
    longitude = session.get('longitude', 'Not provided')
    location = session.get('location', 'Not provided')
    elivation = session.get('elivation', 'Not provided')
    distance = session.get('distance', 'Not provided')
    
    return render_template('location.html', location=location, elivation=elivation, distance=distance, latitude=latitude, longitude=longitude, postal_code=postal_code)


#TODO divide by zero error    
@app.route('/solar', methods=['GET', 'POST'])
def solar():
    # Retrieve form data from session
    surface_area = session.get('surface_area', 'Not provided')
    postal_code = session.get('postal_code', 'Not provided')
    array_type_num = session.get('array_type', 'Not provided')
    module_type_num = session.get('module_type', 'Not provided')
    tilt = session.get('tilt', 'Not provided')
    
    array_types = {
        '0': 'Fixed Carport',
        '1': 'Fixed - Roof Mounted',
        '2': '1-Axis Tracking',
        '3': '1-Axis Backtracking',
        '4': '2-Axis',
    }
    
    
    module_types = {
        '0': 'Standard',
        '1': 'Premium',
        '2': 'Thin Film',
    }
    
    module_efficiencies = {
    'Standard': 21.7,
    'Premium': 22.26,
    'Thin Film': 19.3
    }
    
    # convert the array_type and module type back to strings
    array_type_name = array_types.get(array_type_num, 'Not provided')
    module_type_name = module_types.get(module_type_num, 'Not provided')
    
    # Call PV Watts API
    # Calculate system capacity
    efficiency = module_efficiencies.get(module_type_name, 'Unknown efficiency')
  
    conversion_factor = 1  # [KW / m^2]
    system_capacity_kW = float(surface_area) * efficiency/100 * conversion_factor # kW
    system_capacity_W = system_capacity_kW*1000
    total_dc_yearly = 0
    total_ac_yearly = 0

    #array_type_num = array_type_options[array_type]
    #module_type_num = module_type_options[module_type]

    # Define the URL for the PVWatts V8 API
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"

    # Specify the parameters for the API request
    params = {
        "api_key": "rS4jhBrbjOjG2Rs1d2PZD6HGaIvO1gjDofyabEOV",
        "azimuth": 180,
        "system_capacity": system_capacity_kW,
        "losses": 14.0,
        "array_type": array_type_num,
        "module_type": module_type_num,
        "gcr": 0.4,
        "dc_ac_ratio": 1.2,
        "inv_eff": 98.0, # inverter efficeny 
        "radius": 0,
        "timeframe": 'hourly',
        "tilt": float(tilt),
        "address": postal_code,
        
    }

    # Make the GET request to the PVWatts API
    response = requests.get(url, params=params)
    
    
    # Default values for variables
    hours = []
    ac_hourly = 0
    dc_hourly = 0
    ac_monthly = []
    dc_monthly = []
    poa_monthly = []
    solrad_monthly = []
    temp_cell_monthly = []
    temp_ambient_monthly = []
    latitude = longitude = location_string = elivation = distance_from_site = 'Not available'
    total_dc_yearly = total_ac_yearly = 0

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Display outputs
        ac_hourly = data['outputs']['ac']
        dc_hourly = data['outputs']['dc']
        # Assuming ac_hourly contains 24 hours of data for a single day
        hours = [i for i in range(len(ac_hourly))]

        # If ac_hourly contains data for multiple days, you might want to create a more complex structure
        # For example, if ac_hourly represents a week of hourly data (24*7=168 hours)
        day_time_hours = [f"Day {i//24 + 1}, Hour {i%24}" for i in range(len(ac_hourly))]

        ac_monthly = data["outputs"]["ac_monthly"]
        poa_monthly = data["outputs"]["poa_monthly"]
        solrad_monthly = data["outputs"]["solrad_monthly"]
        dc_monthly = data["outputs"]["dc_monthly"]
        temp_cell_monthly = data["outputs"]["tcell"]
        temp_ambient_monthly = data["outputs"]["tamb"]
        station_info = data['station_info']
        
        latitude = station_info["lat"]
        longitude = station_info["lon"]
        location_string = str(station_info["city"])+', '+str(station_info["state"])+', '+str(station_info["country"])
        elivation = station_info['elev'] #[m]
        distance_from_site = station_info['distance']
        
        session['latitude'] = latitude
        session['longitude'] = longitude
        session['postal_code'] = postal_code
        session['location'] = location_string
        session['elivation'] = elivation
        session['distance'] = distance_from_site

        total_dc_yearly = np.sum(dc_monthly) #[kWh DC]
        total_ac_yearly = np.sum(ac_monthly) #[kWh AC]
    else:
        # Error
        print(f"Error: Received status code {response.status_code}")
        print(response.text)  # This might provide more details on the error
        
    # Sample data
    months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

    # Line chart
    fig1 = px.line(x=months, y=ac_monthly, title="Monthly Solar Power AC Generation", labels={"x": "Month", "y": "Power (kW)"})
    
    # Scatter plot
    fig2 = px.line(x=months, y=dc_monthly, title="Monthly Solar Power DC Generation", labels={"x": "Month", "y": "Power (kW)"})
    
    fig3 = px.line(x=months, y=solrad_monthly, title="Solar Radiance Average Monthly", labels={"x": "Month", "y": "Input Radiation Input Radiation (kWh/m²)"})

    # Update traces to change line color to yellow and add other styling
    fig3.update_traces(line=dict(color='yellow', width=2), mode='lines+markers', marker=dict(color='DarkSlateGrey', size=8, line=dict(width=2, color='DarkSlateGrey')))

    # Plot average monthly cell temperature
    monthly_averages_cell = [sum(temp_cell_monthly[i:i+730])/730 for i in range(0, len(temp_cell_monthly), 730)]
    fig4 = px.bar(x=months, y=monthly_averages_cell, title="Average Monthly Cell Temperature", labels={"x": "Month", "y": "Temperature (°C)"}, color_discrete_sequence=['orange'])
    
    monthly_averages_ambient = [sum(temp_ambient_monthly[i:i+730])/730 for i in range(0, len(temp_ambient_monthly), 730)]
    fig5 = px.bar(x=months, y=monthly_averages_ambient, title="Average Monthly Ambient Temperature", labels={"x": "Month", "y": "Temperature (°C)"}, color_discrete_sequence=['blue'])

    
    # Convert plots to HTML
    plot1 = fig1.to_html(full_html=False)
    plot2 = fig2.to_html(full_html=False)
    plot3 = fig3.to_html(full_html=False)
    plot4 = fig4.to_html(full_html=False)
    plot5 = fig5.to_html(full_html=False)
    
    # adjust solar cost based on choice
    # premium solar pannel
    if module_type_num == 1:
        solar_cost_per_watt = 0.45 # $[CAD] / W 
        pannel_wattage = 575 #W
    # standard pannel
    elif module_type_num == 0:
        solar_cost_per_watt = 0.56
        pannel_wattage = 550 #W
    # thin film
    else:
        solar_cost_per_watt = 1.54
        pannel_wattage = 110 #W
    
    # calculate number of solar pannels and the cost associated
    solar_cost = system_capacity_W * solar_cost_per_watt # $ [CAD]
    
    # calculate the cost of mounts
    costCarport = 1.57  # approximate $/W per Hayter Group
    costRoof = 0.410 # approximate $/W per Hayter Group
    
    # ADJUST MOUNT COST BASED ON TYPE in $/W
    # fixed carport
    if array_type_num == 0:
        cost_mount = costCarport
    # fixed roof mounted
    elif array_type_num == 1:
        # otherwise
        cost_mount = costRoof
    # 1 axis tracking
    elif array_type_num ==2 or array_type_num ==3:
        cost_mount = 0.410
    # otherwise 2 axis tracking
    else:
        cost_mount = 1.14
    
    cost_mount_total = cost_mount*system_capacity_W
    total_solar_installed_cost = solar_cost+cost_mount_total
    
    # calculate the number of pannels
    num_pannels = system_capacity_W / pannel_wattage

    ####### Finical Plots ########
    projLife = 30 # years
    #degredation_percent = 0.4 # yearly percent output decrease
    #total_degredation = degredation_percent*projLife 
    avg_degredation = 0.935
    generation_array = np.array([1/3, 1/2, 2/3, 1, 1.2, 1.4, 1.6, 2]) * total_ac_yearly
    buyback_pricing = np.array([0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4])
    upfront_cost = total_solar_installed_cost
    
    # Calculating yearly revenue for each generation scenario
    yearly_revenue = np.outer(generation_array, buyback_pricing)
    
    total_project_rev_with_degredation = yearly_revenue*avg_degredation*projLife
   
    # Calculate ROI for each generation and buyback pricing
    roi = np.zeros_like(total_project_rev_with_degredation)
    non_zero_cost = upfront_cost != 0
    if non_zero_cost:
        roi[non_zero_cost] = (total_project_rev_with_degredation[non_zero_cost] - upfront_cost) / upfront_cost * 100

    # Calculate Payback Period
    payback_period = np.full_like(yearly_revenue, np.inf)  # Use infinity to indicate undefined or infinite payback period
    non_zero_revenue = yearly_revenue * avg_degredation != 0
    payback_period[non_zero_revenue] = upfront_cost / (yearly_revenue[non_zero_revenue] * avg_degredation)

    min_buyback_total_rev = total_project_rev_with_degredation[3][0]
    total_project_profit_w_degredation = min_buyback_total_rev - upfront_cost
    min_buyback_roi = roi[3][0]
    min_buyback_payback_period = payback_period[3][0]
    
    
    # Creating Plotly plot
    data = []
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=yearly_revenue[i], mode='lines', name=f'Yearly Gen: {gen:.2f} kWh')
        data.append(trace)

    layout = go.Layout(
        title='Yearly Revenue for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='Revenue Yearly (CAD)'),
        legend=dict(title='Yearly Generation'),
    )

    fig6 = go.Figure(data=data, layout=layout)

    # Encoding plot to HTML
    solar_rev_plot = fig6.to_html(full_html=False)
    
    
    # Create empty list to store Plotly traces
    traces = []

    # Loop through each generation scenario and create a trace
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=roi[i], mode='lines', name=f'Yearly Gen: {generation_array[i]:.2f} kWh')
        traces.append(trace)

    # Define layout
    layout = go.Layout(
        title='ROI including degredation for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='ROI [%]'),
        legend=dict(title='Yearly Generation')
    )

    # Create figure and add traces
    fig7 = go.Figure(data=traces, layout=layout)
    # Encoding plot to HTML
    solar_ROI_plot = fig7.to_html(full_html=False)
    
    # Create empty list to store Plotly traces
    traces = []

    # Loop through each generation scenario and create a trace
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=payback_period[i], mode='lines', name=f'Yearly Gen: {generation_array[i]:.2f} kWh')
        traces.append(trace)

    # Define layout
    layout = go.Layout(
        title='Payback Period including degredation for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='Payback Period (Years)'),
        legend=dict(title='Yearly Generation')
    )

    # Create figure and add traces
    fig8 = go.Figure(data=traces, layout=layout)
    solar_payback_plot = fig8.to_html(full_html=False)

    
    # profit calcs
    # File paths for each year's CSV data
    file_paths = {
        '2023': './static\Pricing_Data\PUB_PriceHOEPPredispOR_2023_v393.csv'
    }
    
    # Combine all years of data into a single DataFrame
    all_data = pd.DataFrame()
    for file_path in file_paths.values():
        yearly_data = pd.read_csv(file_path, skiprows=2)
        yearly_data.columns = ['Date', 'Hour', 'HOEP', 'Hour 1 Predispatch', 'Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 10 Min non-sync', 'OR 30 Min']
        yearly_data = yearly_data[['Date', 'Hour', 'HOEP']]
        yearly_data = yearly_data[yearly_data['Hour'].apply(lambda x: x.isnumeric())]
        yearly_data['HOEP'] = pd.to_numeric(yearly_data['HOEP'], errors='coerce')
        all_data = pd.concat([all_data, yearly_data])

    # Clean the data and reset index
    all_data.reset_index(drop=True, inplace=True)
    all_data.dropna(subset=['HOEP'], inplace=True)
    
    # Load the price data
    hourly_prices = all_data['HOEP'].values  # $ /MWh / h Make sure to define 'all_data' with hourly price data
    hourly_price_per_Wh = hourly_prices/1000000 # $[CAD] / W (in one hour)
    hourly_profit = ac_hourly * hourly_price_per_Wh
    
    total_solar_revenue = np.sum(hourly_profit)

    hourly_solar_revenue_fig = go.Figure(data=[
        go.Bar(x=hours, y=hourly_profit, text=hourly_profit, textposition='auto')
    ])
    
    # Update the layout of the plot
    hourly_solar_revenue_fig.update_layout(
        title='Hourly Profit from Selling Solar Energy at HOEP for 1 Year',
        xaxis_title='Hour of the Day',
        yaxis_title='Profit ($)',
        plot_bgcolor='white'
    )

    # Convert the figure to HTML for rendering in Flask
    hourly_solar_revenue_plot = hourly_solar_revenue_fig.to_html(full_html=False)
    
    # update session with costs
    session['solar_pannel_cost'] = solar_cost
    session['solar_mount_cost'] = cost_mount_total
    session['solar_total_cost'] = total_solar_installed_cost
    
    # update session with revenue including degredation based on $0.1/kWh
    session['solar_project_revenue'] = min_buyback_total_rev
    session['solar_project_payback_period'] = min_buyback_payback_period
    session['solar_project_roi'] = min_buyback_roi
    # project profit
    session['solar_project_profit'] = total_project_profit_w_degredation
    
    # Commit session after updating
    session.modified = True
    
    return render_template(
        'solar.html', 
        total_project_profit_w_degredation=total_project_profit_w_degredation,
        min_buyback_roi = min_buyback_roi,
        min_buyback_payback_period = min_buyback_payback_period,
        min_buyback_total_rev=min_buyback_total_rev,
        total_solar_revenue=total_solar_revenue,
        hourly_solar_revenue_plot=hourly_solar_revenue_plot,
        solar_ROI_plot=solar_ROI_plot,
        solar_payback_plot=solar_payback_plot,
        solar_rev_plot=solar_rev_plot,
        total_solar_installed_cost=total_solar_installed_cost,
        cost_mount=cost_mount_total,
        num_pannels=num_pannels, 
        solar_cost=solar_cost, 
        surface_area=surface_area, 
        postal_code=postal_code, 
        array_type=array_type_name, 
        module_type=module_type_name, 
        tilt=tilt, 
        system_capacity=system_capacity_kW, 
        total_dc_yearly=total_dc_yearly, 
        total_ac_yearly=total_ac_yearly, 
        plot1=plot1, 
        plot2=plot2, 
        plot3=plot3, 
        plot4=plot4, 
        plot5=plot5
        )


@app.route('/contact')
def contact():
    return render_template('contact.html')



@app.route('/wind', methods=['GET', 'POST'])
def wind():
    # fetch user input
    # Retrieve latitude and longitude from session
    latitude = session.get('latitude', 'Not provided')
    longitude = session.get('longitude', 'Not provided')
    postal_code = session.get('postal_code', 'Not provided')
    num_turbines = session.get('num_turbines', 'Not provided')
    turbine_height = session.get('turbineHeight', 'Not provided')

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    hours = []
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "hourly": ["temperature_2m", "relative_humidity_2m", "surface_pressure", "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
   

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_relative_humidity_2m = hourly.Variables(1).ValuesAsNumpy()
    hourly_surface_pressure = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(3).ValuesAsNumpy()
    hourly_wind_speed_100m = hourly.Variables(4).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(5).ValuesAsNumpy()
    hourly_wind_direction_100m = hourly.Variables(6).ValuesAsNumpy()
    
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True) - pd.Timedelta(seconds=1),  # Adjust end time
            freq=pd.Timedelta(seconds=hourly.Interval())
        )
    }

    hourly_data["temperature_2m"] = hourly_temperature_2m
    hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
    hourly_data["surface_pressure"] = hourly_surface_pressure
    hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
    hourly_data["wind_speed_100m"] = hourly_wind_speed_100m
    hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
    hourly_data["wind_direction_100m"] = hourly_wind_direction_100m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    
    
    # Convert wind speeds from km/h to m/s
    hourly_data["wind_speed_10m"] = hourly_data["wind_speed_10m"] / 3.6
    hourly_data["wind_speed_100m"] = hourly_data["wind_speed_100m"] / 3.6

    # Add the converted wind speeds to the DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)

    # Convert 'date' to a datetime index in the DataFrame
    hourly_dataframe.set_index('date', inplace=True)

    ### Montly Average Plotting ###
    
    # Extract month from the index (date)
    hourly_dataframe['month'] = hourly_dataframe.index.month

    # Group by month and calculate the mean wind speed for plotting
    monthly_wind_speed_10m = hourly_dataframe.groupby('month')['wind_speed_10m'].mean()
    monthly_wind_speed_100m = hourly_dataframe.groupby('month')['wind_speed_100m'].mean()
    
    
    # Create a new DataFrame for Plotly
    monthly_wind_speed_df = pd.DataFrame({
        'Month': monthly_wind_speed_10m.index,
        'Wind Speed at 10m (m/s)': monthly_wind_speed_10m.values,
        'Wind Speed at 100m (m/s)': monthly_wind_speed_100m.values
    })

    # Plotting with Plotly Express
    fig1 = px.line(monthly_wind_speed_df, x='Month', y=['Wind Speed at 10m (m/s)', 'Wind Speed at 100m (m/s)'],
                  labels={'value': 'Wind Speed (m/s)', 'variable': 'Measurement'}, title='Monthly Average Wind Speeds')

    # Heights and average wind speeds for curve fitting
    heights = np.array([10, 100])  # Heights in meters
    average_wind_speeds = np.array([
        monthly_wind_speed_10m.mean(),
        monthly_wind_speed_100m.mean()
    ])

    # Define the logarithmic wind profile function
    def log_wind_profile(z, a, b):
        return a + b * np.log(z)

    # Perform curve fitting
    try:
        popt, pcov = curve_fit(log_wind_profile, heights, average_wind_speeds)
        
        # Generate heights for the fitted curve visualization
        fitted_heights = np.linspace(10, 100, 100)  # From 10m to 100m
        fitted_speeds = log_wind_profile(fitted_heights, *popt)

        # Create Plotly figure for the fitted curve and original data
        fig2 = px.scatter(x=heights, y=average_wind_speeds, labels={'x': 'Height (m)', 'y': 'Wind Speed (m/s)'}, title='Wind Speed vs. Height with Logarithmic Fit')
        fig2.add_scatter(x=fitted_heights, y=fitted_speeds, mode='lines', name=f'Fitted: v = {popt[0]:.2f} + {popt[1]:.2f} * ln(z)')
        
    except Exception as e:
        print("An error occurred during curve fitting:", e)
        fig2 = None
    
    # Convert Plotly figures to HTML
    plot1 = fig1.to_html(full_html=False) if 'fig1' in locals() else None
    plot2 = fig2.to_html(full_html=False) if fig2 else None
    
    ### Monthly Specific Plotting ###
    # Sample wind speeds at 10m and 100m for demonstration
    # These should be replaced with your actual data arrays
    wind_speed_10m = hourly_data["wind_speed_10m"]
    wind_speed_100m = hourly_data["wind_speed_100m"]
    
    # Define the heights for the wind speeds we have and the ones we want to interpolate
    measured_heights = np.array([10, 100])
    interpolate_heights = np.array([18, 24, 30, 36, 55, 80])
    interpolated_speeds = {f'wind_speed_{h}m': [] for h in interpolate_heights}

    # Iterate over each hour
    for i in range(len(wind_speed_10m)):
        # Current wind speeds at 10m and 100m for the hour
        current_speeds = np.array([wind_speed_10m[i], wind_speed_100m[i]])
        
        # Fit the curve for this hour's data
        popt, _ = curve_fit(log_wind_profile, measured_heights, current_speeds)
        
        # Use the obtained fit to calculate speeds at desired heights
        for h in interpolate_heights:
            interpolated_speed = log_wind_profile(h, *popt)
            interpolated_speeds[f'wind_speed_{h}m'].append(interpolated_speed)
            
    for height, speeds in interpolated_speeds.items():
        hourly_data[height] = speeds

    # Convert hourly_data to a DataFrame
    hourly_dataframe = pd.DataFrame(data=hourly_data)
    
    
    # Check if 'date' is already the index or a column in the DataFrame
    if 'date' in hourly_dataframe.columns:
        # Convert 'date' column to datetime if it exists as a column
        hourly_dataframe['date'] = pd.to_datetime(hourly_dataframe['date'])
        # Set 'date' as the DataFrame index
        hourly_dataframe.set_index('date', inplace=True)
    elif not isinstance(hourly_dataframe.index, pd.DatetimeIndex):
        # If 'date' is not a column and the index is not already a DatetimeIndex, attempt conversion
        hourly_dataframe.index = pd.to_datetime(hourly_dataframe.index)

    ## make the user select a month defualt it to JAN
    selected_month = 1  # Default to January
    ## Get user request ##
    if request.method == 'POST':
        selected_month = int(request.form.get('month_select', selected_month))
    
    # Filter the DataFrame for January
    month_data = hourly_dataframe[hourly_dataframe.index.month == selected_month]
    
    
    # Ensure january_data's index is in a compatible format (datetime)
    dates = month_data.index  # No need to convert if index is already datetime

    # Create a subplot
    fig3 = make_subplots(rows=1, cols=1)

    # Plotting measured wind speeds at 10m and 100m
    fig3.add_trace(go.Scatter(x=dates, y=month_data['wind_speed_10m'], mode='lines', name='10m (measured)', line=dict(color='blue')))
    fig3.add_trace(go.Scatter(x=dates, y=month_data['wind_speed_100m'], mode='lines', name='100m (measured)', line=dict(color='red')))

    # Plotting interpolated wind speeds
    for height in ['18m', '24m', '30m', '36m', '55m', '80m']:
        fig3.add_trace(go.Scatter(x=dates, y=month_data[f'wind_speed_{height}'], mode='lines', name=f'{height} (interpolated)', line=dict(dash='dash')))


    
    # Update plot layout
    fig3.update_layout(
        title='Hourly Wind Speeds for January by Height',
        xaxis_title='Date',
        yaxis_title='Wind Speed (m/s)',
        legend_title='Wind Speed',
        xaxis=dict(
            tickangle=45,
            nticks=10  # Optional: Adjust the number of x-axis labels
        )
    )
    
    # update month name title
    month_name = calendar.month_name[selected_month]
    months = {i: calendar.month_name[i] for i in range(1, 13)}

    fig3.update_layout(title=f'Hourly Wind Speeds for {month_name} by Height')

    # Convert Plotly figure to HTML for Flask rendering
    hourly_month_plot_html = fig3.to_html(full_html=False)
    
    capacity_factor = 0.90 # [%] efficiency rating
    capicity_per_turbine = 100 #[kW]
    total_system_capacity_kw = num_turbines*capacity_factor*capicity_per_turbine
    
    
    ############### hourly averaged #################
    # Ensure 'hour' column is correct based on the 'date' index
    hourly_dataframe['hour'] = hourly_dataframe.index.hour
    # Specify the wind speed heights you have in your DataFrame
    wind_speed_heights = ['wind_speed_10m','wind_speed_55m', 'wind_speed_100m']

    # Initialize a dictionary to hold our summary stats for each height
    hourly_stats = {height: {'mean': [], 'std': []} for height in wind_speed_heights}

    # Calculate mean and std for each height and hour
    for height in wind_speed_heights:
        for hour in range(24):
            # Filter data for the current hour and height
            hourly_data = hourly_dataframe[hourly_dataframe['hour'] == hour][height]
            # Calculate mean and std, append to the respective lists
            hourly_stats[height]['mean'].append(hourly_data.mean())
            hourly_stats[height]['std'].append(hourly_data.std())

    # Initialize a new Plotly figure for the hourly wind speed statistics
    stat_fig = Figure()

    # Add traces for each height with error bars
    for height in wind_speed_heights:
        stat_fig.add_trace(
            Scatter(
                x=list(range(24)),
                y=hourly_stats[height]['mean'],
                error_y=dict(
                    type='data', # value of error bar given in data coordinates
                    array=hourly_stats[height]['std'],
                    visible=True
                ),
                name=height
            )
        )

    # Update layout of the figure
    stat_fig.update_layout(
        title='Average Hourly Wind Speed for Each Height with Error Bars',
        xaxis_title='Hour of Day',
        yaxis_title='Wind Speed (m/s)',
        legend_title='Height',
        xaxis=dict(tickvals=list(range(24)), ticktext=[f"{h}:00" for h in range(24)])
    )

    # Convert the figure to HTML
    stat_plot_html = stat_fig.to_html(full_html=False)
    
    ######## Wind Generation ##########
    # see wind speed output relationship code to see how values were derived
    # funtion to take in houly wind speed and calculate the kW produced in that hour
    def wind_output_fit(x_wind_speed):
        # Use a vectorized approach to handle an array of wind speeds
        output = np.where((x_wind_speed > 11) | (x_wind_speed < 3), 0, 51 + (119 / np.pi) * np.arctan(0.8 * x_wind_speed - 6))
        return output

    # Apply the function to calculate the power output for each hour
    specific_wind_speeds = hourly_dataframe['wind_speed_'+str(turbine_height)+'m']
    hourly_dataframe['power_output'] = wind_output_fit(specific_wind_speeds)
    
    average_yearly_speed = specific_wind_speeds.mean()

    # Assume `num_turbines` is defined somewhere in your code
    # Calculate total power generation for each hour
    hourly_dataframe['total_power_gen'] = hourly_dataframe['power_output'] * num_turbines * capacity_factor

    # Sum up to find the total yearly generation
    total_yearly_generation = hourly_dataframe['total_power_gen'].sum()

    # Aggregate this hourly data by month
    monthly_generation = hourly_dataframe.resample('M').sum()['total_power_gen']

    # Plot the monthly generation
    wind_month_fig = go.Figure()
    wind_month_fig.add_trace(go.Bar(x=monthly_generation.index.month, y=monthly_generation.values))

    # Update plot layout
    wind_month_fig.update_layout(
        title='Monthly Wind Power Generation',
        xaxis_title='Month',
        yaxis_title='Total Power Generation (kWh)',
        xaxis=dict(tickvals=list(range(1, 13)), ticktext=list(calendar.month_name[1:])),
        yaxis=dict(title='Total Power Generation (kWh)')
    )

    # You can convert this figure to HTML for Flask as before
    monthly_gen_plot_html = wind_month_fig.to_html(full_html=False)
    
    ## calculate upfront cost for wind ##
    cost_per_turbine = 165709.29 #[CAD]
    wind_cost = num_turbines*cost_per_turbine


    ####### Finical Plots ########
    projLife = 30 # years
    degredation_percent = 0.6 # percent output decrease per year
    total_degredation = degredation_percent*projLife #percent decrease after projectlife
    avg_degredation = (100-total_degredation/2)/100
    
    generation_array = np.array([1/3, 1/2, 2/3, 1, 1.2, 1.4, 1.6, 2]) * total_yearly_generation
    buyback_pricing = np.array([0.1, 0.13, 0.16, 0.19, 0.22, 0.25, 0.28, 0.31, 0.34, 0.37, 0.4])
    upfront_cost = wind_cost
    
    # Calculating yearly revenue for each generation scenario
    yearly_revenue = np.outer(generation_array, buyback_pricing)
    
    total_project_rev_with_degredation = yearly_revenue*avg_degredation*projLife

    # Calculate ROI for each generation and buyback pricing
    roi = np.zeros_like(total_project_rev_with_degredation)
    non_zero_cost = upfront_cost != 0
    if non_zero_cost:
        roi[non_zero_cost] = (total_project_rev_with_degredation[non_zero_cost] - upfront_cost) / upfront_cost * 100

    # Calculate Payback Period
    payback_period = np.full_like(yearly_revenue, np.inf)  # Use infinity to indicate undefined or infinite payback period
    non_zero_revenue = yearly_revenue * avg_degredation != 0
    payback_period[non_zero_revenue] = upfront_cost / (yearly_revenue[non_zero_revenue] * avg_degredation)
    
    min_buyback_total_rev = total_project_rev_with_degredation[3][0]
    total_project_profit_w_degredation = min_buyback_total_rev-upfront_cost
    min_buyback_roi = roi[3][0]
    min_buyback_payback_period = payback_period[3][0]
    
    # Creating Plotly plot
    data = []
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=yearly_revenue[i], mode='lines', name=f'Yearly Gen: {gen:.2f} kWh')
        data.append(trace)

    layout = go.Layout(
        title='Yearly Revenue for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='Yearly Revenue (CAD)'),
        legend=dict(title='Yearly Generation'),
    )

    fig6 = go.Figure(data=data, layout=layout)

    # Encoding plot to HTML
    wind_rev_plot = fig6.to_html(full_html=False)
    
    
    # Create empty list to store Plotly traces
    traces = []

    # Loop through each generation scenario and create a trace
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=roi[i], mode='lines', name=f'Yearly Gen: {generation_array[i]:.2f} kWh')
        traces.append(trace)

    # Define layout
    layout = go.Layout(
        title='ROI for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='ROI [%]'),
        legend=dict(title='Yearly Generation')
    )

    # Create figure and add traces
    fig7 = go.Figure(data=traces, layout=layout)
    # Encoding plot to HTML
    wind_ROI_plot = fig7.to_html(full_html=False)
    
    # Create empty list to store Plotly traces
    traces = []

    # Loop through each generation scenario and create a trace
    for i, gen in enumerate(generation_array):
        trace = go.Scatter(x=buyback_pricing, y=payback_period[i], mode='lines', name=f'Yearly Gen: {generation_array[i]:.2f} kWh')
        traces.append(trace)

    # Define layout
    layout = go.Layout(
        title='Payback Period for Different Yearly Generations Compared to Buyback Pricing',
        xaxis=dict(title='Buyback Pricing (CAD/kWh)'),
        yaxis=dict(title='Payback Period (Years)'),
        legend=dict(title='Yearly Generation')
    )

    # Create figure and add traces
    fig8 = go.Figure(data=traces, layout=layout)
    wind_payback_plot = fig8.to_html(full_html=False)


    # profit calcs
    # File paths for each year's CSV data
    file_paths = {
        '2023': './static/Pricing_Data/PUB_PriceHOEPPredispOR_2023_v393.csv'
    }
    
    # Combine all years of data into a single DataFrame
    all_data = pd.DataFrame()
    for file_path in file_paths.values():
        yearly_data = pd.read_csv(file_path, skiprows=2)
        yearly_data.columns = ['Date', 'Hour', 'HOEP', 'Hour 1 Predispatch', 'Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 10 Min non-sync', 'OR 30 Min']
        yearly_data = yearly_data[['Date', 'Hour', 'HOEP']]
        yearly_data = yearly_data[yearly_data['Hour'].apply(lambda x: x.isnumeric())]
        yearly_data['HOEP'] = pd.to_numeric(yearly_data['HOEP'], errors='coerce')
        all_data = pd.concat([all_data, yearly_data])

    # Clean the data and reset index
    all_data.reset_index(drop=True, inplace=True)
    all_data.dropna(subset=['HOEP'], inplace=True)
    
    # Load the price data
    hourly_prices = all_data['HOEP'].values  # $ /MWh / h Make sure to define 'all_data' with hourly price data
    hourly_price_per_kWh = hourly_prices/1000 # $[CAD] / kW (in one hour)
    hourly_wind_gen = hourly_dataframe['total_power_gen']
    hourly_revenue_wind = hourly_wind_gen*hourly_price_per_kWh # $[CAD] / h
    
    hours = [i for i in range(len(hourly_wind_gen))]
    total_wind_revenue = np.sum(hourly_revenue_wind)

    hourly_wind_revenue_fig = go.Figure(data=[
        go.Bar(x=hours, y=hourly_revenue_wind, text=hourly_revenue_wind, textposition='auto')
    ])
    
    # Update the layout of the plot
    hourly_wind_revenue_fig.update_layout(
        title='Hourly Profit from Selling Wind Energy at HOEP for 1 Year',
        xaxis_title='Hour of the Day',
        yaxis_title='Profit ($)',
        plot_bgcolor='white'
    )
    
    hourly_wind_revenue_plot = hourly_wind_revenue_fig.to_html(full_html=False)

    # update session with costs
    session['wind_upfront_cost'] = wind_cost
    
    # update session with revenue including degredation based on $0.1/kWh
    session['wind_project_revenue'] = min_buyback_total_rev
    session['wind_project_payback_period'] = min_buyback_payback_period
    session['wind_project_roi'] = min_buyback_roi
    # project profit
    session['wind_project_profit'] = total_project_profit_w_degredation
    
    # Commit session after updating
    session['wind_visited'] = True
    session.modified = True

    # Pass all plots to the template
    return render_template(
        'wind.html',
        total_project_profit_w_degredation=total_project_profit_w_degredation,
        min_buyback_roi = min_buyback_roi,
        min_buyback_payback_period = min_buyback_payback_period,
        min_buyback_total_rev=min_buyback_total_rev,
        total_wind_revenue=total_wind_revenue,
        hourly_wind_revenue_plot=hourly_wind_revenue_plot,
        wind_ROI_plot=wind_ROI_plot,
        wind_rev_plot=wind_rev_plot,
        wind_payback_plot=wind_payback_plot,
        average_yearly_speed=average_yearly_speed,
        wind_cost=wind_cost,
        total_yearly_generation=total_yearly_generation,
        monthly_gen_plot_html=monthly_gen_plot_html,
        turbine_height=turbine_height,
        total_system_capacity_kw=total_system_capacity_kw,
        num_turbines=num_turbines,
        months=months,
        hourly_wind_plot=hourly_month_plot_html,
        stat_plot=stat_plot_html, # pass the new plot to the template
        plot1=plot1,
        plot2=plot2,
        postal_code=postal_code,
        latitude=latitude,
        longitude=longitude
    )


@app.route('/battery', methods=['GET','POST'])
def battery():
    battery_consumption = session.get('battery_consumption', 'Not provided') or 0
    battery_runtime = session.get('battery_runtime', 'Not provided') or 0
    
    # warrenty for Megapack cover 15 years representing ~70% which correlates to about 5500
    # therefore set the max at slightly less than double
    battery_final_percent = session.get('battery_final_percent') or 50
    battery_max_cycles = int(session.get('battery_max_cycles') or 100000)
    
    battery_final_percent = float(battery_final_percent)/100
    
    
    battery_capacity = battery_consumption*battery_runtime
    
    # cost
    battery_cost_per_kWh = 748.2439907 # [CAD / kWh]
    total_upfront_battery_cost = battery_capacity * battery_cost_per_kWh
    
    # The directory where your CSV files are stored
    folder_path = './static/Battery-cycles'

    # Get all CSV files in the folder
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

    # Define specific colors for each SoC type
    color_map = {
        '100to25': 'rgb(255, 0, 0)',  # Red
        '100to40': 'rgb(0, 128, 0)',  # Green
        '100to50': 'rgb(0, 0, 255)',  # Blue
        '75to25': 'rgb(255, 165, 0)', # Orange
        '75to65': 'rgb(128, 0, 128)', # Purple
        '85to25': 'rgb(255, 192, 203)'# Pink
    }

    # Create a Plotly figure
    fig = go.Figure()

    # Iterate through each file and plot the data
    for file_path in file_paths:
        # Read the CSV file
        data = pd.read_csv(file_path, header=None)  # Assuming no header in the CSV files
        # Convert to numpy array 
        data_array = np.asarray(data)

        # Sort the data by the first column (cycle number) for accurate interpolation
        sorted_indices = np.argsort(data_array[:, 0])
        sorted_data = data_array[sorted_indices]

        # Create a linear interpolation function
        interpolation_func = interp1d(sorted_data[:, 0], sorted_data[:, 1], kind='linear')

        # Generate new x values for plotting the interpolation
        new_x = np.linspace(sorted_data[0, 0], sorted_data[-1, 0], num=500)
        new_y = interpolation_func(new_x)

        # Plot the data
        label = os.path.basename(file_path).replace('.csv', '')
        # Choose a color for the current data set
        color_css = color_map.get(label, 'rgb(0,0,0)')
        fig.add_trace(go.Scatter(x=sorted_data[:, 0], y=sorted_data[:, 1], mode='markers', name=f'{label} Data', marker_color=color_css))
        fig.add_trace(go.Scatter(x=new_x, y=new_y, mode='lines', name=f'{label} Interpolation', line_color=color_css, showlegend=False))

    # Update the layout of the plot
    fig.update_layout(
        title='Battery Capacity Retention over Cycle Number with Linear Interpolation at 20 deg Celsius',
        xaxis_title='Cycle Number',
        yaxis_title='Capacity Retention (%)',
        legend_title='Legend'
    )

    # Show the plot
    # Convert the figure to HTML and embed it
    capacity_plot = fig.to_html(full_html=False)
    
    # get the HEOP pricing data
    # File paths for each year's CSV data
    file_paths = {
        '2023': './static/Pricing_Data/PUB_PriceHOEPPredispOR_2023_v393.csv'
    }
    
    # Combine all years of data into a single DataFrame
    all_data = pd.DataFrame()
    for file_path in file_paths.values():
        yearly_data = pd.read_csv(file_path, skiprows=2)
        yearly_data.columns = ['Date', 'Hour', 'HOEP', 'Hour 1 Predispatch', 'Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 10 Min non-sync', 'OR 30 Min']
        yearly_data = yearly_data[['Date', 'Hour', 'HOEP']]
        yearly_data = yearly_data[yearly_data['Hour'].apply(lambda x: x.isnumeric())]
        yearly_data['HOEP'] = pd.to_numeric(yearly_data['HOEP'], errors='coerce')
        all_data = pd.concat([all_data, yearly_data])

    # Clean the data and reset index
    all_data.reset_index(drop=True, inplace=True)
    all_data.dropna(subset=['HOEP'], inplace=True)
    
    # Load the price data
    hourly_prices = all_data['HOEP'].values  # Make sure to define 'all_data' with hourly price data

    # Directory containing degradation models
    folder_path = './static/Battery-cycles'
    file_paths = glob.glob(os.path.join(folder_path, '*.csv'))
    
    initial_capacity = battery_capacity/1000
    charge_rate = initial_capacity  # MW/h based off of the 2hr charge discharge rate
    discharge_rate = initial_capacity  # MW/h  
    
    # Create a Plotly figure
    year_one_cap_fig = go.Figure()

    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        sorted_data = data.sort_values(by=0)
        cycle_numbers = sorted_data[0].values
        capacities = sorted_data[1].values / 100

        capacity_func = interp1d(cycle_numbers, capacities, bounds_error=False, fill_value="extrapolate")
        basename = os.path.basename(file_path).replace('.csv', '')
        max_charge_level, min_charge_level = map(int, basename.split('to'))
        min_charge_level /= 100
        max_charge_level /= 100

        tracked_capacities = simulate_and_track_capacity(capacity_func, hourly_prices, initial_capacity, charge_rate, discharge_rate, min_charge_level, max_charge_level)

        # Generate x values (time in hours)
        hours = np.linspace(0, len(tracked_capacities), num=len(tracked_capacities))
        
        color_css = color_map.get(basename, 'rgb(0,0,0)')
        # Add trace to Plotly figure
        year_one_cap_fig.add_trace(go.Scatter(x=hours, y=tracked_capacities, mode='lines', name=f'{basename} (SoC: {max_charge_level*100}% to {min_charge_level*100}%)', line_color=color_css))

    # Update the layout of the Plotly plot
    year_one_cap_fig.update_layout(
        title='Battery Capacity Over each hour of Year 1 for Different SoC Types',
        xaxis_title='Time (Hours)',
        yaxis_title='Battery Capacity (MWh)',
        legend_title='SoC Types'
    )

    # Convert the figure to HTML for rendering in Flask
    year_one_cap_plot = year_one_cap_fig.to_html(full_html=False)
    
    # battery revenue
    
    
    # Simulate and compare results for each degradation model
    profits = {}
    cycles_used = {}
    
    hours_per_year = 24 * 365  # Total hours in a year

    
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        sorted_data = data.sort_values(by=0)
        cycle_numbers = sorted_data[0].values
        capacities = sorted_data[1].values / 100  # Convert percentages to fraction of initial capacity

        capacity_func = interp1d(cycle_numbers, capacities, bounds_error=False, fill_value="extrapolate")
        basename = os.path.basename(file_path).replace('.csv', '')
        target_capacity = battery_final_percent * initial_capacity

        profit, cycles = simulate_daily_cycling(capacity_func, hourly_prices, initial_capacity, target_capacity, battery_max_cycles)
        profits[basename] = profit
        cycles_used[basename] = cycles
    
    cap_fig_all_time = go.Figure()
    for file_path in file_paths:
        data = pd.read_csv(file_path, header=None)
        sorted_data = data.sort_values(by=0)
        cycle_numbers = sorted_data[0].values
        capacities = sorted_data[1].values / 100 # Scale to percentage

        capacity_func = interp1d(cycle_numbers, capacities, bounds_error=False, fill_value='extrapolate')
        basename = os.path.basename(file_path).replace('.csv', '')
        cycles = cycles_used[basename]  # Ensure this data is available from previous simulations

        hours = np.arange(0, cycles * 24, 24)
        capacities = capacity_func(hours / 24) * initial_capacity
        years = hours / hours_per_year
        
        color_css = color_map.get(basename, 'rgb(0,0,0)')
        cap_fig_all_time.add_trace(go.Scatter(x=years, y=capacities, mode='lines', name=basename, line_color=color_css))

    cap_fig_all_time.update_layout(
        title='Comparison of Battery Capacity Degradation Over Time for Different SoCs',
        xaxis_title='Time (Years)',
        yaxis_title='Battery Capacity (MWh)',
        legend_title="SoC Types"
    )

    cap_plot_all_time = cap_fig_all_time.to_html(full_html=False)
    
    # Now integrate the new profit plot:
    # Extract SoC types, corresponding profits, and cycles used for plotting
    soc_types = list(profits.keys())
    profit_values = list(profits.values())
    cycles_for_soc = [cycles_used[soc] for soc in soc_types]

    # Create a Plotly bar chart for profits
    profit_fig = go.Figure()

    profit_fig.add_trace(go.Bar(
        x=soc_types,
        y=profit_values,
        text=[f"{cycles} cycles" for cycles in cycles_for_soc],
        marker_color='skyblue',  # Uniform color or you could use a color scale
        hoverinfo='y+text'
    ))

    profit_fig.update_layout(
        title='Total Lifetime Revenue Comparison by Degradation Model with Cycle Count',
        xaxis_title='State of Charge Type',
        yaxis_title='Total Revenue ($)',
        legend_title="Legend",
        xaxis={'tickangle': 45}  # Rotate labels for better legibility
    )

    # Convert the figure to HTML for rendering in Flask
    profit_plot_html = profit_fig.to_html(full_html=False)
    
    # Find the minimum profit and its index
    min_profit = float(min(profit_values))
    min_profit_index = profit_values.index(min_profit)

    # Find the SoC type associated with the minimum profit
    min_profit_soc_type = soc_types[min_profit_index]
    
    amount_earned_min = min_profit-total_upfront_battery_cost
    
    # update session with costs
    session['battery_upfront_cost'] = total_upfront_battery_cost
    # update session with revenue including degredation based on 25to100 
    session['battery_project_revenue'] = min_profit
    # project profit
    session['battery_project_profit'] = amount_earned_min
    
    session['battery_visited'] = True
    session.modified = True
    
    return render_template(
        'battery.html', 
        battery_capacity=battery_capacity, 
        battery_runtime=battery_runtime, 
        battery_final_percent=battery_final_percent,
        battery_max_cycles=battery_max_cycles,
        battery_consumption=battery_consumption, 
        total_upfront_battery_cost=total_upfront_battery_cost,
        capacity_plot=capacity_plot, 
        year_one_cap_plot=year_one_cap_plot,
        cap_plot_all_time=cap_plot_all_time,
        profit_plot_html=profit_plot_html,
        min_profit=min_profit,
        min_profit_soc_type=min_profit_soc_type,
        amount_earned_min=amount_earned_min
    )
 
# Function to simulate energy arbitrage and track capacity
def simulate_and_track_capacity(capacity_func, prices, initial_capacity, charge_rate, discharge_rate, min_charge_level, max_charge_level):
    energy_stored = 0
    cycle_count = 0
    capacities = [initial_capacity]  # Track capacity over time

    for i in range(1, len(prices)):
        current_capacity = capacity_func(cycle_count) * initial_capacity
        min_allowed_energy = current_capacity * min_charge_level
        max_allowed_energy = current_capacity * max_charge_level
        
        predicted_next_price = prices[i]
        current_price = prices[i-1]

        if current_price < predicted_next_price and energy_stored < max_allowed_energy:
            energy_to_charge = min(charge_rate, max_allowed_energy - energy_stored)
            energy_stored += energy_to_charge
            cycle_count += 1
        elif current_price > predicted_next_price and energy_stored > min_allowed_energy:
            energy_to_sell = min(discharge_rate, energy_stored - min_allowed_energy)
            energy_stored -= energy_to_sell
            cycle_count += 1

        capacities.append(current_capacity)

    return capacities

# Function to simulate energy arbitrage and track capacity
def simulate_daily_cycling(capacity_func, prices, initial_capacity, target_capacity, max_cycles):
    daily_profit = 0
    cycle_count = 0
    days = len(prices) // 24  # Assuming prices are hourly and we have complete days
    current_capacity = initial_capacity

    while current_capacity > target_capacity and cycle_count < max_cycles:
        for day in range(days):
            if current_capacity <= target_capacity or cycle_count >= max_cycles:
                break  # Stop the simulation if capacity drops to the target or max cycles reached
            daily_prices = prices[day*24:(day+1)*24]
            if len(daily_prices) < 24:
                continue  # Skip incomplete days

            min_price_hour = np.argmin(daily_prices)
            max_price_hour = np.argmax(daily_prices)

            current_capacity = capacity_func(cycle_count) * initial_capacity
            buy_price = daily_prices[min_price_hour]
            sell_price = daily_prices[max_price_hour]

            daily_profit += (sell_price - buy_price) * current_capacity
            cycle_count += 1  # Increment cycle count for each full day

    return daily_profit, cycle_count

@app.route('/pricing', methods=['GET','POST'])
def pricing():
    # Load and clean the data
    file_path = './static/Pricing_Data/PUB_PriceHOEPPredispOR_2023_v393.csv'
    data = pd.read_csv(file_path, skiprows=2, header=None)
    data.columns = ['Date', 'Hour', 'HOEP', 'Hour 1 Predispatch', 'Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 10 Min non-sync', 'OR 30 Min']
    data_cleaned = data[['Date', 'Hour', 'HOEP']].dropna()
    data_cleaned = data_cleaned[data_cleaned['Hour'] != 'Hour']
    data_cleaned['Hour'] = data_cleaned['Hour'].astype(int) - 1
    data_cleaned['Datetime'] = pd.to_datetime(data_cleaned['Date']) + pd.to_timedelta(data_cleaned['Hour'], unit='h')
    data_cleaned['HOEP'] = pd.to_numeric(data_cleaned['HOEP'], errors='coerce')
    data_cleaned.dropna(subset=['HOEP'], inplace=True)

    # Plotting with Plotly
    hourly_price_fig = px.line(data_cleaned, x='Datetime', y='HOEP', title='Historical Hourly Energy Prices (HOEP) for 2023', labels={'HOEP': 'HOEP for 1 MWh (in $)'})
    hourly_price_fig.update_xaxes(tickangle=45)
    
    hourly_price_2023= data_cleaned['HOEP']
    average_2023 = np.mean(hourly_price_2023)

    # Convert plot to HTML
    hourly_price_plot = hourly_price_fig.to_html(full_html=False)
    
    # Define the file paths
    file_paths = {
        '2021': './static/Pricing_Data/PUB_PriceHOEPPredispOR_2021_v395.csv',
        '2022': './static/Pricing_Data/PUB_PriceHOEPPredispOR_2022_v396.csv',
        '2023': './static/Pricing_Data/PUB_PriceHOEPPredispOR_2023_v393.csv'
    }

    hourly_avg_all_years = []
    traces = []

    for year, file_path in file_paths.items():
        data = pd.read_csv(file_path, skiprows=2)
        data.columns = ['Date', 'Hour', 'HOEP', 'Hour 1 Predispatch', 'Hour 2 Predispatch', 'Hour 3 Predispatch', 'OR 10 Min Sync', 'OR 10 Min non-sync', 'OR 30 Min']
        data = data[['Date', 'Hour', 'HOEP']].dropna()
        data = data[data['Hour'].apply(lambda x: x.isnumeric())]
        data['Hour'] = data['Hour'].astype(int) - 1
        data['HOEP'] = pd.to_numeric(data['HOEP'], errors='coerce').dropna()

        hourly_avg = data.groupby('Hour')['HOEP'].mean().reset_index()
        hourly_avg_all_years.append(hourly_avg.set_index('Hour'))

        trace = go.Scatter(x=hourly_avg['Hour'], y=hourly_avg['HOEP'], mode='lines', name=f'Average HOEP {year}')
        traces.append(trace)

    combined_hourly_avg = pd.concat(hourly_avg_all_years, axis=1)
    combined_hourly_mean = combined_hourly_avg.mean(axis=1)
    combined_hourly_std = combined_hourly_avg.std(axis=1)

    trace_combined = go.Scatter(x=combined_hourly_mean.index, y=combined_hourly_mean, mode='lines+markers', name='Total Average HOEP',
                                 error_y=dict(type='data', array=combined_hourly_std, visible=True))
    traces.append(trace_combined)

    # Define the layout
    layout = go.Layout(title='Average Hourly Energy Prices (HOEP) for Each Hour Over Years with Error Bars',
                       xaxis=dict(title='Hour of the Day'),
                       yaxis=dict(title='Average HOEP (in $)'),
                       showlegend=True)

    # Create figure and convert to HTML
    hourly_avg_price_fig = go.Figure(data=traces, layout=layout)
    hourly_avg_price_plot = hourly_avg_price_fig.to_html(full_html=False)
    
    

    return render_template('pricing.html', average_2023=average_2023, hourly_price_plot=hourly_price_plot, hourly_avg_price_plot=hourly_avg_price_plot)


@app.route('/download-csv')
def download_csv():
    data = [
        ["Parameter", "Value"],
        ["System Capacity (kW)", session.get('system_capacity', 'Not provided')],
        ["Total Yearly DC (kWh)", session.get('total_dc_yearly', 'Not provided')],
        ["Total Yearly AC (kWh)", session.get('total_ac_yearly', 'Not provided')],
        # Add more data rows as needed
    ]

    # Create a generator to stream the CSV data
    def generate_csv():
        line = csv.writer(sys.stdout)
        for row in data:
            yield ','.join(row) + '\n'

    # Return the streamed response
    return Response(generate_csv(), mimetype='text/csv', headers={"Content-Disposition": "attachment; filename=solar_data.csv"})


@app.route('/reset')
def reset_session():
    for key in [ 'wind_visited', 'battery_visited']:
        session.pop(key, None)
    return redirect(url_for('home'))


@app.route('/summary', methods=['GET','POST'])
def summary():
    
    # Check all necessary data are available
    required_keys = ['wind_upfront_cost', 'solar_pannel_cost']
    if not all(key in session for key in required_keys):
        return 'Data processing incomplete', 400  # or redirect to an error page or the start of the flow
    
    # Wind
    # update session with costs
    wind_upfront_cost= session.get('wind_upfront_cost', 0)
    #wind_upfront_cost = float(wind_upfront_cost)
    # update session with revenue including degredation based on $0.1/kWh
    wind_project_revenue = session.get('wind_project_revenue', 0)
    
    wind_project_payback_period= session.get('wind_project_payback_period', 0)
    wind_project_roi = session.get('wind_project_roi', 0)
    # project profit
    wind_project_profit = session.get('wind_project_profit', 0)
    
    
    # Solar
    # update session with costs
    solar_pannel_cost = session.get('solar_pannel_cost', 0)
    solar_mount_cost = session.get('solar_mount_cost', 0)
    solar_total_cost = session.get('solar_total_cost', 0)
    
    # update session with revenue including degredation based on $0.1/kWh
    solar_project_revenue = session.get('solar_project_revenue', 0)
    solar_project_payback_period = session.get('solar_project_payback_period', 0)
    solar_project_roi = session.get('solar_project_roi', 0)
    # project profit
    solar_project_profit = session.get('solar_project_profit', 0)
    
    # Battery
    battery_upfront_cost = session.get('battery_upfront_cost', 0)
    # update session with revenue including degredation based on 25to100 
    battery_project_revenue = session.get('battery_project_revenue', 0)
    # project profit
    battery_project_profit = session.get('battery_project_profit', 0)
    
    # Total Project Cost
    total_project_costs = battery_upfront_cost + solar_total_cost + wind_upfront_cost
    # Total project Revenue
    total_project_revenue = battery_project_revenue + solar_project_revenue + wind_project_revenue
    # Total Project Profit
    total_project_profit = total_project_revenue - total_project_costs
    
    # total project lifetime
    total_lifetime = 30 # years
    
    # total project ROI
    total_roi = total_project_profit/total_project_costs * 100 # [%] return
    
    # adjusted payback period based on fluctauting yearly revenues and degredation
    total_payback_period = total_project_costs/(total_project_revenue/total_lifetime)
    
    
    return render_template(
        'summary.html',
        total_project_costs=total_project_costs,
        total_project_revenue=total_project_revenue,
        total_project_profit=total_project_profit,
        total_lifetime=total_lifetime,
        total_roi=total_roi,
        total_payback_period=total_payback_period,
        battery_upfront_cost=battery_upfront_cost,
        battery_project_revenue=battery_project_revenue,
        battery_project_profit=battery_project_profit,
        wind_upfront_cost=wind_upfront_cost, 
        wind_project_revenue=wind_project_revenue,
        wind_project_payback_period=wind_project_payback_period,
        wind_project_roi=wind_project_roi,
        wind_project_profit=wind_project_profit,
        solar_pannel_cost=solar_pannel_cost,
        solar_mount_cost=solar_mount_cost,
        solar_total_cost=solar_total_cost,
        solar_project_revenue=solar_project_revenue,
        solar_project_payback_period=solar_project_payback_period,
        solar_project_roi=solar_project_roi,
        solar_project_profit=solar_project_profit
    )

if __name__ == '__main__':
    app.run(debug=True)
