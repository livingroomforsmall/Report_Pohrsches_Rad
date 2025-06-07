import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# x und y-Daten
import numpy as np

x = np.array([0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0], dtype=float)

# y-Werte zuerst als Liste definieren, dann Umrechnung mit NumPy
raw_y = np.array([20.0, 14.0, 9.6, 6.6, 4.6, 3.0, 2.0, 1.4, 0.8, 0.5], dtype=float)
y = raw_y / 25 * np.pi  # Umrechnung in Radiant


# Modellfunktion: y(x) = A * exp(Bx) 
def model(x, A, B):
    return A * np.exp(B * x)

# Curve fitting: optimiert A, B
try:
    # Verbesserte Startwerte und Grenzen
    params, params_covariance = curve_fit(
        model,
        x,
        y,
        p0=[1, -0.1],  # Angepasste Startwerte

        maxfev=5000  # Erhöhte Anzahl der Funktionsaufrufe
    )
    A, B= params
    print(f"Fit-Parameter: A={A:.4f}, B={B:.4f}")

    # Kovarianzmatrix
    perr = np.sqrt(np.diag(params_covariance))
    print(f"Unsicherheiten der Parameter:")
    print(f"Unsicherheit A: {perr[0]:.4f}")
    print(f"Unsicherheit B: {perr[1]:.4f}")

except RuntimeError as e:
    print(f"Fehler beim Fitten: {e}")
    exit()

# Plot mit halblogarithmischer Darstellung (logarithmische y-Achse)
plt.figure(figsize=(10, 6))

# Halblogarithmischer Plot der Daten
plt.semilogy(x, y, 'o', markersize=3, label="Daten")

# Plot der Fit-Kurve
plt.plot(x, model(x, *params), '-', label=f"Fit: ${A:.4f} e^{{{B:.4f} t}}$")

# Achsenbeschriftungen und Titel
plt.xlabel("Zeit (s)", fontsize=12)
plt.ylabel("Auslenkung (rad)", fontsize=12)

# Gitter und Legende
plt.grid(True)
plt.legend(fontsize=10)

# Layout anpassen und Plot anzeigen
plt.tight_layout()
plt.show()