import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Excel-Datei laden, Tabelle 3
try:
    df = pd.read_excel("Messdaten.xlsx", sheet_name="Tabelle3")
except FileNotFoundError:
    print("Fehler: Die Datei 'Messdaten.xlsx' wurde nicht gefunden.")
    exit()

df = df.replace({',': '.'}, regex=True)

# Daten ab Zeile 40 (Index 48)
df = df.iloc[39:].copy()  # Wichtig: .copy(), um eine Kopie zu erstellen

# First dataset (for 0.2A)
x1 = np.array(df.iloc[:, 18] - 1.83, dtype=float)  # Explizit float
y1 = np.array(df.iloc[:, 19] / 98.15 * np.pi, dtype=float)  # Explizit float
valid_data_mask1 = np.isfinite(x1) & np.isfinite(y1)
x1 = x1[valid_data_mask1]
y1 = y1[valid_data_mask1]

# Second dataset (for 0.3A)
x2 = np.array(df.iloc[:, 21] - 1.83, dtype=float)  # Explizit float
y2 = np.array(df.iloc[:, 22] / 98.15 * np.pi, dtype=float)  # Explizit float
valid_data_mask2 = np.isfinite(x2) & np.isfinite(y2)
x2 = x2[valid_data_mask2]
y2 = y2[valid_data_mask2]

# Modellfunktion: y(x) = A * exp(Bx) * cos(Cx + D)
def model(x, A, B, C, D):
    return A * np.exp(B * x) * np.cos(C * x + D)

# Curve fitting for the first dataset (0.2A)
params1, _ = curve_fit(model, x1, y1, p0=[1, -0.1, 1, 0], maxfev=5000)
A1, B1, C1, D1 = params1

# Curve fitting for the second dataset (0.3A)
params2, _ = curve_fit(model, x2, y2, p0=[1, -0.1, 1, 0], maxfev=5000)
A2, B2, C2, D2 = params2

# Create a 1x2 subplot (two subplots in one row)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # 1 row, 2 columns

# First subplot (for 0.8A)
axes[0].plot(x1, y1, 'o', markersize=3, label="Daten")
axes[0].plot(x1, model(x1, *params1), '-', label=f"Fit: Phi={A1:.2f}exp({B1:.2f}t)cos({C1:.2f}t+{D1:.2f})")
axes[0].set_xlabel("Zeit (s)", fontsize=12)
axes[0].set_ylabel("Auslenkung (rad)", fontsize=12)
axes[0].set_title("Auftragung für 0.8A", fontsize=14)
axes[0].grid(True)
axes[0].legend(fontsize=10)

# Second subplot (for 0.9A)
axes[1].plot(x2, y2, 'o', markersize=3, label="Daten")
axes[1].plot(x2, model(x2, *params2), '-', label=f"Fit: Phi={A2:.2f}exp({B2:.2f}t)cos({C2:.2f}t+{D2:.2f})")
axes[1].set_xlabel("Zeit (s)", fontsize=12)
axes[1].set_ylabel("Auslenkung (rad)", fontsize=12)
axes[1].set_title("Auftragung für 0.9A", fontsize=14)
axes[1].grid(True)
axes[1].legend(fontsize=10)

# Adjust layout for better spacing
plt.tight_layout()

# Show the plot
plt.show()
