import tkinter as tk
import requests
import matplotlib.pyplot as plt
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
        "soiling": "12|4|45|23|9|99|67|12.54|54|9|0|7.6",
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

        # Plotting graphs
        months = [
            'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'
        ]

        plt.figure(figsize=(10, 6))

        plt.subplot(2, 2, 1)
        plt.plot(months, ac_monthly, marker='o')
        plt.title('AC Monthly vs Month')
        plt.xlabel('Month')
        plt.ylabel('AC (kWh)')
        mplcursors.cursor(hover=True)

        plt.subplot(2, 2, 2)
        plt.plot(months, poa_monthly, marker='o')
        plt.title('POA Monthly vs Month')
        plt.xlabel('Month')
        plt.ylabel('POA (kWh/m^2)')
        mplcursors.cursor(hover=True)

        plt.subplot(2, 2, 3)
        plt.plot(months, solrad_monthly, marker='o')
        plt.title('Solar Radiation Monthly vs Month')
        plt.xlabel('Month')
        plt.ylabel('Solar Radiation (kWh/m^2)')
        mplcursors.cursor(hover=True)

        plt.subplot(2, 2, 4)
        plt.plot(months, dc_monthly, marker='o')
        plt.title('DC Monthly vs Month')
        plt.xlabel('Month')
        plt.ylabel('DC (kWh)')
        mplcursors.cursor(hover=True)

        plt.tight_layout()
        plt.show()

    else:
        output_text.config(state=tk.NORMAL)
        output_text.delete(1.0, tk.END)
        output_text.insert(tk.END, f"Error: {response.status_code}")
        output_text.config(state=tk.DISABLED)


# Create the main window
window = tk.Tk()
window.title("Solar Panel Calculator")

# Create labels and entry widgets for user input
surface_area_label = tk.Label(window, text="Surface Area (m^2):")
surface_area_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")
surface_area_entry = tk.Entry(window)
surface_area_entry.grid(row=0, column=1, padx=5, pady=5)

address_label = tk.Label(window, text="Postal Code:")
address_label.grid(row=1, column=0, padx=5, pady=5, sticky="w")
address_entry = tk.Entry(window)
address_entry.grid(row=1, column=1, padx=5, pady=5)

array_type_label = tk.Label(window, text="Array Type (0-5):")
array_type_label.grid(row=2, column=0, padx=5, pady=5, sticky="w")
array_type_entry = tk.Entry(window)
array_type_entry.grid(row=2, column=1, padx=5, pady=5)

module_type_label = tk.Label(window, text="Module Type (0-2):") 
module_type_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")  
module_type_entry = tk.Entry(window)
module_type_entry.grid(row=3, column=1, padx=5, pady=5)  

system_loss_label = tk.Label(window, text="System Loss (%):")
system_loss_label.grid(row=4, column=0, padx=5, pady=5, sticky="w")
system_loss_entry = tk.Entry(window)
system_loss_entry.grid(row=4, column=1, padx=5, pady=5)

tilt_label = tk.Label(window, text="Tilt (0-90 degrees):")
tilt_label.grid(row=5, column=0, padx=5, pady=5, sticky="w")
tilt_entry = tk.Entry(window)
tilt_entry.grid(row=5, column=1, padx=5, pady=5)

calculate_button = tk.Button(window, text="Calculate", command=get_solar_data)
calculate_button.grid(row=6, column=0, columnspan=2, pady=10)

# Start the GUI event loop
window.mainloop()
