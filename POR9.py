import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Daten aus der Tabelle (hier als Beispiel manuell eingefügt, du kannst sie aus einer Datei laden)
lambda_ = np.array([0.05, 0.10, 0.16, 0.25, 0.35, 0.46, 0.59, 0.73, 0.92, 1.12, 1.32, 1.56, 1.82, 2.10])
omega = np.array([3.13, 3.11, 3.11, 3.09, 3.07, 3.08, 3.03, 3.04, 2.99, 2.92, 2.82, 2.68, 2.49, 2.30])

# Modellfunktion: omega = sqrt(A - lambda^2)
def model(lambda_, A):
    return np.sqrt(A - lambda_**2)

# Fit mit Fehlern in den Daten (wir nehmen an, dass keine Fehler angegeben sind)
params, covariance = curve_fit(model, lambda_, omega, p0=[5])  # Startwert für A als 5

# Extrahierter Parameter A
A = params[0]
print(f"Fit-Parameter: A={A:.2f}")

# Erstellen des Plots
plt.figure(figsize=(8, 5))

# Plot der Originaldaten
plt.plot(lambda_, omega, 'o', markersize=6, label="Daten", color='blue')

# Plot des Fits
lambda_fit = np.linspace(min(lambda_), max(lambda_), 100)
omega_fit = model(lambda_fit, *params)
plt.plot(lambda_fit, omega_fit, '-', label=f"Fit: $\omega = \sqrt{{A - \lambda^2}}$ mit $A={A:.2f}$")

# Achsenbeschriftungen und Titel
plt.xlabel("$\lambda$ [s$^{-1}$]", fontsize=12)
plt.ylabel("$\omega$ [s$^{-1}$]", fontsize=12)
plt.title("Fit von $\omega$ gegen $\lambda$", fontsize=14)

# Legende und Gitter
plt.legend(fontsize=10)
plt.grid(True)

# Plot anzeigen
plt.tight_layout()
plt.show()
