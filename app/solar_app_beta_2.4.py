import tkinter as tk
from tkinter import ttk
import requests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mplcursors
import numpy as np
from scipy.stats import linregress

plot_canvas = None
solar_tab = None

def setup_solar_tab(notebook):
    global plot_canvas_frame, plot_canvas, solar_tab  # Declare as global if you need to access it outside this function

    # Create the solar_tab before referencing it
    solar_tab = ttk.Frame(notebook)
    notebook.add(solar_tab, text='Solar')

    # Create a Canvas within the solar tab for scrollability
    plot_canvas = tk.Canvas(solar_tab)
    plot_scrollbar = ttk.Scrollbar(solar_tab, orient="vertical", command=plot_canvas.yview)
    plot_canvas.configure(yscrollcommand=plot_scrollbar.set)

    plot_scrollbar.pack(side=tk.RIGHT, fill="y")
    plot_canvas.pack(side=tk.LEFT, fill="both", expand=True)

    # Frame within the Canvas to hold plots
    plot_canvas_frame = ttk.Frame(plot_canvas)  # This is the correct initialization
    canvas_window = plot_canvas.create_window((10, 50), window=plot_canvas_frame, anchor='nw', tags='plot_frame')

    plot_canvas_frame.bind("<Configure>", lambda event, canvas=plot_canvas: canvas.configure(scrollregion=plot_canvas.bbox("all")))

def get_input_data(plot_canvas):

    # Get the inputs from the GUI
    surface_area = float(surface_area_entry.get())
    address = address_entry.get()
    array_type = array_type_combobox.current() 
    module_type = module_type_combobox.current() 
    system_loss = float(system_loss_entry.get())
    tilt = int(tilt_entry.get())

    # Map the selected index to the actual values
    array_type_options = [0, 1, 2, 3, 4]
    module_type_options = [0, 1, 2]

    array_type = array_type_options[array_type]
    module_type = module_type_options[module_type]

    # Calculate system capacity
    input_cell_efficiency = 21.3  # [%]
    conversion_factor = 1  # [KW / m^2]
    system_capacity = surface_area * input_cell_efficiency * conversion_factor

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

    if response.status_code == 200:
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

        fig, axs = plt.subplots(8, 1, figsize=(15, 50))
        #fig.subplots_adjust(hspace=0.5)
        
        axs[0].plot(months, ac_monthly, marker='o')
        axs[0].set_title('AC Monthly vs Month')
        axs[0].set_xlabel('Month')
        axs[0].set_ylabel('AC (kWh)')
        axs[0].margins(y=0.05)
        axs[0].grid(True)

        axs[1].plot(months, poa_monthly, marker='o')
        axs[1].set_title('POA Monthly vs Month')
        axs[1].set_xlabel('Month')
        axs[1].set_ylabel('POA (kWh/m^2)')
        axs[1].grid(True)

        axs[2].plot(months, solrad_monthly, marker='o')
        axs[2].set_title('input Radiation Monthly vs Month')
        axs[2].set_xlabel('Month')
        axs[2].set_ylabel('input Radiation (kWh/m^2)')
        axs[2].grid(True)

        axs[3].plot(months, dc_monthly, marker='o')
        axs[3].set_title('DC Monthly vs Month')
        axs[3].set_xlabel('Month')
        axs[3].set_ylabel('DC (kWh)')
        axs[3].grid(True)

        # Plot average monthly cell temperature
        monthly_averages_cell = [sum(temp_cell_monthly[i:i+730])/730 for i in range(0, len(temp_cell_monthly), 730)]
        axs[4].bar(months, monthly_averages_cell, color='blue')
        axs[4].set_title('Average Monthly Cell Temperature')
        axs[4].set_xlabel('Month')
        axs[4].set_ylabel('Temperature (C)')
        axs[4].grid(True)

        # Plot average monthly ambient temperature
        monthly_averages_ambient = [sum(temp_ambient_monthly[i:i+730])/730 for i in range(0, len(temp_ambient_monthly), 730)]
        axs[5].bar(months, monthly_averages_ambient, color='green')
        axs[5].set_title('Average Monthly Ambient Temperature')
        axs[5].set_xlabel('Month')
        axs[5].set_ylabel('Temperature (C)')
        axs[5].grid(True)

        # Create scatterplot for POA irradiance vs DC output
        axs[6].scatter(poa_monthly, dc_monthly, color='orange', label='POA vs DC')
        slope, intercept, r_value, p_value, std_err = linregress(poa_monthly, dc_monthly)
        axs[6].plot(poa_monthly, intercept + slope * np.array(poa_monthly), color='blue', label='Line of Best Fit')
        axs[6].set_title('POA Irradiance vs DC Output')
        axs[6].set_xlabel('POA (kWh/m^2)')
        axs[6].set_ylabel('DC (kWh)')
        axs[6].legend()
        axs[6].grid(True)

        # Create scatterplot for POA irradiance vs AC output
        axs[7].scatter(poa_monthly, ac_monthly, color='orange', label='POA vs AC')
        slope, intercept, r_value, p_value, std_err = linregress(poa_monthly, ac_monthly)
        axs[7].plot(poa_monthly, intercept + slope * np.array(poa_monthly), color='blue', label='Line of Best Fit')
        axs[7].set_title('POA Irradiance vs AC Output')
        axs[7].set_xlabel('POA (kWh/m^2)')
        axs[7].set_ylabel('AC (kWh)')
        axs[7].legend()
        axs[7].grid(True)

        for ax in axs:
            mplcursors.cursor(hover=True)

        # Clear previous plots if any
        for widget in plot_canvas_frame.winfo_children():
            widget.destroy()

        # Embed Matplotlib figures into Canvas
        canvas = FigureCanvasTkAgg(fig, master=plot_canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        # Update the scroll region of the canvas
        plot_canvas.configure(scrollregion=plot_canvas.bbox("all"))

        calculate_button.configure(bg='green')

        notebook.select(solar_tab)  

    else:
        print(f"Error: {response.status_code}")

def get_battery_data():
    pass

def get_wind_data():
    pass

def get_financial_data():
    pass

# Create the main window
window = tk.Tk()
window.title("Renewable Energy Calculator")

# Calculate 80% of the screen width and height
screen_width = window.winfo_screenwidth()
screen_height = window.winfo_screenheight()
desired_width = int(0.8 * screen_width)
desired_height = int(0.8 * screen_height)

# Set window geometry to desired width and height
window.geometry(f"{desired_width}x{desired_height}")

notebook = ttk.Notebook(window)
notebook.pack(fill='both', expand=True)

# Inputs tab
inputs_tab = ttk.Frame(notebook)
notebook.add(inputs_tab, text='Inputs')

# Create labels and entry widgets for input input
input_input_frame = tk.Frame(inputs_tab)
input_input_frame.pack(side=tk.LEFT, padx=10, pady=10, anchor='nw')

surface_area_label = tk.Label(input_input_frame, text="Surface Area (m^2):")
surface_area_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
surface_area_entry = tk.Entry(input_input_frame)
surface_area_entry.grid(row=0, column=1, padx=5, pady=5)

address_label = tk.Label(input_input_frame, text="Postal Code:")
address_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
address_entry = tk.Entry(input_input_frame)
address_entry.grid(row=1, column=1, padx=5, pady=5)

array_type_label = tk.Label(input_input_frame, text="Array Type:")
array_type_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
array_type_options = ['Fixed - Open Rack', 'Fixed - Roof Mounted', '1-Axis', '1-Axis Backtracking', '2-Axis']
array_type_combobox = ttk.Combobox(input_input_frame, values=array_type_options)
array_type_combobox.grid(row=2, column=1, padx=5, pady=5)

module_type_label = tk.Label(input_input_frame, text="Module Type:")
module_type_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")
module_type_options = ['Standard', 'Premium', 'Thin film']
module_type_combobox = ttk.Combobox(input_input_frame, values=module_type_options)
module_type_combobox.grid(row=3, column=1, padx=5, pady=5)

system_loss_label = tk.Label(input_input_frame, text="System Loss (%):")
system_loss_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
system_loss_entry = tk.Entry(input_input_frame)
system_loss_entry.grid(row=4, column=1, padx=5, pady=5)

tilt_label = tk.Label(input_input_frame, text="Tilt (0-90 degrees):")
tilt_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
tilt_entry = tk.Entry(input_input_frame)
tilt_entry.grid(row=5, column=1, padx=5, pady=5)

calculate_button = tk.Button(input_input_frame, text="Calculate", command=lambda: get_input_data(plot_canvas))
calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

# Solar panel tab for plots
# Setup a canvas frame inside solar_tab for plots
setup_solar_tab(notebook)  # Call the setup function for the solar tab


# Other tabs...
battery_tab = ttk.Frame(notebook)
notebook.add(battery_tab, text='Battery')
# Setup for battery_tab...

wind_tab = ttk.Frame(notebook)
notebook.add(wind_tab, text='Wind')
# Setup for wind_tab...

financial_tab = ttk.Frame(notebook)
notebook.add(financial_tab, text='Financial')
# Setup for financial_tab...

# Start the GUI event loop
window.mainloop()