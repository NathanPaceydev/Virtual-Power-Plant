from flask import Flask, render_template, request, redirect, url_for, session, Response
import requests
import sys
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.templates.default = "none"
import csv

app = Flask(__name__)

app.secret_key = 'your_secret_key_here'


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Store form data in session
        session['surface_area'] = request.form.get('surfaceArea')
        session['postal_code'] = request.form.get('postalCode')
        session['array_type'] = request.form.get('arrayType')
        session['module_type'] = request.form.get('moduleType')
        session['tilt'] = request.form.get('tilt')

        # Redirect to the solar page
        return redirect(url_for('solar'))

    return render_template('home.html')

@app.route('/solar')
def solar():
    # Retrieve form data from session
    surface_area = session.get('surface_area', 'Not provided')
    postal_code = session.get('postal_code', 'Not provided')
    array_type = session.get('array_type', 'Not provided')
    module_type = session.get('module_type', 'Not provided')
    tilt = session.get('tilt', 'Not provided')
    
    # Call PV Watts API
    # Calculate system capacity
    solar_cell_efficiency = 21.3  # [%] Effeciency of Bi-facial premium cells
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
        "array_type": array_type,
        "module_type": module_type,
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


    return render_template('solar.html', surface_area=surface_area, postal_code=postal_code, array_type=array_type, module_type=module_type, tilt=tilt, system_capacity=system_capacity, total_dc_yearly=total_dc_yearly, total_ac_yearly=total_ac_yearly, plot1=plot1, plot2=plot2, plot3=plot3, plot4=plot4, plot5=plot5)


@app.route('/contact')
def contact():
    return render_template('contact.html')


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
