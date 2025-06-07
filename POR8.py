import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Daten aus der Tabelle (hier als Beispiel manuell eingefügt, du kannst sie aus einer Datei laden)
I = np.array([0.200, 0.300, 0.400, 0.499, 0.600, 0.700, 0.800, 0.894, 1.002, 1.099, 1.199, 1.299, 1.396, 1.502])
lambda_ = np.array([0.05, 0.10, 0.16, 0.25, 0.35, 0.46, 0.59, 0.73, 0.92, 1.12, 1.32, 1.56, 1.82, 2.10])
I_error = np.array([0.005]*len(I))  # Fehler in I, hier konstant, kann je nach Bedarf angepasst werden

# Quadratische Modellfunktion: y(x) = ax^2
def quadratic_model(x, a):
    return a * x**2

# Fit mit Fehlern in den Daten (gewichteter Fit)
params, covariance = curve_fit(quadratic_model, I, lambda_, sigma=I_error, absolute_sigma=True)

# Extrahierte Parameter
a = float(params[0])  # Extrahieren und als float umwandeln
print(f"Fit-Parameter: a={a:.4f}")

# Erstellen des Plots
plt.figure(figsize=(8, 5))

# Plot der Originaldaten mit Fehlerbalken
plt.errorbar(I, lambda_, yerr=I_error, fmt='o', markersize=6, label="Daten", color='blue')

# Plot des Fits
I_fit = np.linspace(min(I), max(I), 100)
lambda_fit = quadratic_model(I_fit, *params)
plt.plot(I_fit, lambda_fit, '-', label=f"Fit: ${a:.2f}I^2$")

# Achsenbeschriftungen und Titel
plt.xlabel("$I$ [A]", fontsize=12)
plt.ylabel("$\lambda$ [s$^{-1}$]", fontsize=12)
plt.title("Fit von $\lambda$ gegen $I$", fontsize=14)

# Legende und Gitter
plt.legend(fontsize=10)
plt.grid(True)

# Plot anzeigen
plt.tight_layout()
plt.show()
