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
from plotly.subplots import make_subplots

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Store existing form data in session
        session['surface_area'] = request.form.get('surfaceArea')
        session['postal_code'] = request.form.get('postalCode')
        session['array_type'] = request.form.get('arrayType')
        session['module_type'] = request.form.get('moduleType')
        session['tilt'] = request.form.get('tilt')

        # Store the new input for the number of wind turbines
        session['num_turbines'] = request.form.get('numTurbines', type=int)  # Default to integer type

        # Redirect to the solar page
        return redirect(url_for('solar'))

    return render_template('home.html')


@app.route('/solar', methods=['GET', 'POST'])
def solar():
    # Retrieve form data from session
    surface_area = session.get('surface_area', 'Not provided')
    postal_code = session.get('postal_code', 'Not provided')
    array_type_num = session.get('array_type', 'Not provided')
    module_type_num = session.get('module_type', 'Not provided')
    tilt = session.get('tilt', 'Not provided')
    
    array_types = {
        '0': 'Fixed Open Rack',
        '1': 'Fixed - Roof Mounted',
        '2': '1-Axis Tracking',
        '3': '1-Axis Backtracking',
        '4': '2-Axis',
    }
    
    module_types = {
        '1': 'Standard',
        '2': 'Premium',
        '3': 'Thin Film',
    }
    
    # Call PV Watts API
    # Calculate system capacity
    solar_cell_efficiency = 22.26  # [%] Effeciency of Bi-facial premium cells
    conversion_factor = 1  # [KW / m^2]
    system_capacity = float(surface_area) * solar_cell_efficiency * conversion_factor
    
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
        "system_capacity": system_capacity,
        "losses": 14.3,
        "array_type": array_type_num,
        "module_type": module_type_num,
        "gcr": 0.4,
        "dc_ac_ratio": 1.2,
        "inv_eff": 96.0,
        "radius": 0,
        "timeframe": 'hourly',
        "dataset": "nsrdb",
        "tilt": float(tilt),
        "address": postal_code,
        "albedo": 0.3,
        "bifaciality": 0.7
    }

    # Make the GET request to the PVWatts API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()

        # Display outputs
        ac_monthly = data["outputs"]["ac_monthly"]
        poa_monthly = data["outputs"]["poa_monthly"]
        solrad_monthly = data["outputs"]["solrad_monthly"]
        dc_monthly = data["outputs"]["dc_monthly"]
        temp_cell_monthly = data["outputs"]["tcell"]
        temp_ambient_monthly = data["outputs"]["tamb"]
        station_info = data['station_info']
        
        latitude = station_info["lat"]
        longitude = station_info["lon"]
        
        session['latitude'] = latitude
        session['longitude'] = longitude
        session['postal_code'] = postal_code

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
    
    
    # convert the array_type and module type back to strings
    array_type_name = array_types.get(array_type_num, 'Not provided')
    module_type_name = module_types.get(module_type_num, 'Not provided')

    return render_template('solar.html', surface_area=surface_area, postal_code=postal_code, array_type=array_type_name, module_type=module_type_name, tilt=tilt, system_capacity=system_capacity, total_dc_yearly=total_dc_yearly, total_ac_yearly=total_ac_yearly, plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4, plot5=plot5)


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
        
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    
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
    interpolate_heights = np.array([20, 30, 40, 50, 60, 80])
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
    for height in ['20m', '30m', '40m', '50m', '60m', '80m']:
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
    
    
    # Pass the plot to the template
    return render_template('wind.html', num_turbines=num_turbines, months=months, hourly_wind_plot=hourly_month_plot_html, plot1=plot1, plot2=plot2, postal_code=postal_code, latitude=latitude, longitude=longitude)


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


if __name__ == '__main__':
    app.run(debug=True)
