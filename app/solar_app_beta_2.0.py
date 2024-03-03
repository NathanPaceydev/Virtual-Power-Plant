import tkinter as tk
from tkinter import ttk
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors

def get_solar_data():
    # Get the inputs from the GUI
    surface_area = float(surface_area_entry.get())
    address = address_entry.get()
    array_type = int(array_type_entry.get())
    module_type = int(module_type_entry.get())  
    system_loss = float(system_loss_entry.get())
    tilt = int(tilt_entry.get())

    # Calculate system capacity
    solar_cell_efficiency = 21.3  # [%]
    conversion_factor = 1  # [KW / m^2]
    system_capacity = surface_area * solar_cell_efficiency * conversion_factor

    # Define the URL for the PVWatts V8 API
    url = "https://developer.nrel.gov/api/pvwatts/v8.json"

    # Specify the parameters for the API request
    params = {
        "api_key": "rS4jhBrbjOjG2Rs1d2PZD6HGaIvO1gjDofyabEOV",
        "azimuth": 180,
        "system_capacity": system_capacity,
        "losses": system_loss,
        "array_type": array_type,
        "module_type": module_type,  
        "gcr": 0.4,
        "dc_ac_ratio": 1.2,
        "inv_eff": 96.0,
        "radius": 0,
        "timeframe": 'hourly',
        "dataset": "nsrdb",
        "tilt": tilt,
        "address": address,
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

        # Plotting graphs
        months = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]

        fig, axs = plt.subplots(6, 1, figsize=(8, 40))

        axs[0].plot(months, ac_monthly, marker='o')
        axs[0].set_title('AC Monthly vs Month')
        axs[0].set_xlabel('Month')
        axs[0].set_ylabel('AC (kWh)')
        axs[0].margins(y=0.05)

        axs[1].plot(months, poa_monthly, marker='o')
        axs[1].set_title('POA Monthly vs Month')
        axs[1].set_xlabel('Month')
        axs[1].set_ylabel('POA (kWh/m^2)')

        axs[2].plot(months, solrad_monthly, marker='o')
        axs[2].set_title('Solar Radiation Monthly vs Month')
        axs[2].set_xlabel('Month')
        axs[2].set_ylabel('Solar Radiation (kWh/m^2)')

        axs[3].plot(months, dc_monthly, marker='o')
        axs[3].set_title('DC Monthly vs Month')
        axs[3].set_xlabel('Month')
        axs[3].set_ylabel('DC (kWh)')

        # Plot average monthly cell temperature
        monthly_averages_cell = [sum(temp_cell_monthly[i:i+730])/730 for i in range(0, len(temp_cell_monthly), 730)]
        axs[4].bar(months, monthly_averages_cell, color='blue')
        axs[4].set_title('Average Monthly Cell Temperature')
        axs[4].set_xlabel('Month')
        axs[4].set_ylabel('Temperature (C)')

        # Plot average monthly ambient temperature
        monthly_averages_ambient = [sum(temp_ambient_monthly[i:i+730])/730 for i in range(0, len(temp_ambient_monthly), 730)]
        axs[5].bar(months, monthly_averages_ambient, color='green')
        axs[5].set_title('Average Monthly Ambient Temperature')
        axs[5].set_xlabel('Month')
        axs[5].set_ylabel('Temperature (C)')

        for ax in axs:
            mplcursors.cursor(hover=True)

        # Embed Matplotlib figures into Canvas
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Attach scrollbar to the canvas
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.get_tk_widget().yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        canvas.get_tk_widget().configure(yscrollcommand=scrollbar.set)

    else:
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Error: {response.status_code}")
        output_text.config(state=tk.DISABLED)


# Create the main window
window = tk.Tk()
window.title("Solar Panel Calculator")

# Create labels and entry widgets for user input
input_frame = tk.Frame(window)
input_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

surface_area_label = tk.Label(input_frame, text="Surface Area (m^2):")
surface_area_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
surface_area_entry = tk.Entry(input_frame)
surface_area_entry.grid(row=0, column=1, padx=5, pady=5)

address_label = tk.Label(input_frame, text="Postal Code:")
address_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
address_entry = tk.Entry(input_frame)
address_entry.grid(row=1, column=1, padx=5, pady=5)

array_type_label = tk.Label(input_frame, text="Array Type (0-5):")
array_type_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
array_type_entry = tk.Entry(input_frame)
array_type_entry.grid(row=2, column=1, padx=5, pady=5)

module_type_label = tk.Label(input_frame, text="Module Type (0-2):") 
module_type_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")  
module_type_entry = tk.Entry(input_frame)
module_type_entry.grid(row=3, column=1, padx=5, pady=5)  

system_loss_label = tk.Label(input_frame, text="System Loss (%):")
system_loss_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
system_loss_entry = tk.Entry(input_frame)
system_loss_entry.grid(row=4, column=1, padx=5, pady=5)

tilt_label = tk.Label(input_frame, text="Tilt (0-90 degrees):")
tilt_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
tilt_entry = tk.Entry(input_frame)
tilt_entry.grid(row=5, column=1, padx=5, pady=5)

calculate_button = tk.Button(input_frame, text="Calculate", command=get_solar_data)
calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

# Create a frame to contain the canvas and scrollbar
canvas_frame = tk.Frame(window)
canvas_frame.grid(row=0, column=1, rowspan=7, padx=10, pady=20, sticky="nsew")

# Configure the grid weights to allow resizing
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

# Start the GUI event loop
window.mainloop()
