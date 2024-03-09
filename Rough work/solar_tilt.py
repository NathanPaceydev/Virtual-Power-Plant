import requests
import matplotlib.pyplot as plt


# Updated Open Rack fixed data with new values
open_rack_deg_updated = [35, 32, 31, 30, 29, 28, 25]
open_rack_production_updated = [1308884, 1312740, 1313300, 1313477, 1313260, 1312642, 1308399]

# Updated Roof Rack fixed data with new values
roof_rack_deg_updated = [40, 37, 36, 35, 33, 32, 31, 28]
roof_rack_production_updated = [1333314, 1336192, 1336509, 1336493, 1335500, 1334509, 1333195, 1327266]

# Plotting Updated Open Rack fixed tilt angle vs Yearly Production
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(open_rack_deg_updated, open_rack_production_updated, marker='o', linestyle='-', color='b')
plt.title('Open Rack Fixed Tilt Angle vs Yearly Production')
plt.xlabel('Tilt Angle (Degrees)')
plt.ylabel('Yearly Production (kWh)')
plt.grid(True)

# Plotting Updated Roof Rack fixed tilt angle vs Yearly Production
plt.subplot(1, 2, 2)
plt.plot(roof_rack_deg_updated, roof_rack_production_updated, marker='o', linestyle='-', color='r')
plt.title('Roof Rack Fixed Tilt Angle vs Yearly Production')
plt.xlabel('Tilt Angle (Degrees)')
plt.ylabel('Yearly Production (kWh)')
plt.grid(True)

plt.tight_layout()
plt.show()
