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

# Daten ab Zeile 49 (Index 48)
df = df.iloc[48:].copy()  # Wichtig: .copy(), um eine Kopie zu erstellen

# x und y-Daten
x = np.array(df.iloc[:, 0] - 2.28, dtype=float)  # Explizit float
y = np.array(df.iloc[:, 1] / 98.15 * np.pi, dtype=float)  # Explizit float

# Modellfunktion: y(x) = A * exp(Bx) * cos(Cx + D)
def model(x, A, B, C, D):
    return A * np.exp(B * x) * np.cos(C * x + D)

# Curve fitting: optimiert A, B, C und D
try:
    # Verbesserte Startwerte und Grenzen
    params, params_covariance = curve_fit(
        model,
        x,
        y,
        p0=[1, -0.1, 1, 0],  # Angepasste Startwerte
        bounds=([-np.inf, -np.inf, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf]),  # Grenzen für C > 0
        maxfev=5000  # Erhöhte Anzahl der Funktionsaufrufe
    )
    A, B, C, D = params
    print(f"Fit-Parameter: A={A:.2f}, B={B:.2f}, C={C:.2f}, D={D:.2f}")
except RuntimeError as e:
    print(f"Fehler beim Fitten: {e}")
    exit()

# Plot
plt.figure(figsize=(10, 6))  # Größere Figur
plt.plot(x, y, 'o', markersize=3, label="Daten")
plt.plot(x, model(x, *params), '-', label=f"Fit: Phi={A:.2f}exp({B:.2f}t)cos({C:.2f}t+{D:.2f})")
plt.xlabel("Zeit (s)", fontsize=12)
plt.ylabel("Auslenkung (rad)", fontsize=12)
plt.title("Auftragung ab Zeile 49 und Fit", fontsize=14)
plt.grid(True)
plt.legend(fontsize=10)
plt.tight_layout()
plt.show()