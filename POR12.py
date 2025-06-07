import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# x und y-Daten
import numpy as np

x = np.array([1.57, 1.76, 1.96, 2.15, 2.37, 2.61, 2.75, 2.94, 3.03, 3.14, 3.23, 3.30, 3.54, 3.95, 4.53], dtype=float)
rawy = np.array([0.95, 1, 1.15, 1.3, 1.6, 2.05, 3, 5.2, 7.75, 8.4, 5.25, 3.55, 2, 1.05, 0.6], dtype=float)
y = rawy - 0.22

def model(x, A, B, C):
    return A / np.sqrt((B**2 - x**2)**2 + 4 * C**2 * x**2)
A0 = max(y) * (9.67)  # A hängt vom Skalierungsfaktor ab
B0 = x[np.argmax(y)]  # Peak-Lage als Näherung für Resonanz
C0 = 0.18  # Dämpfung

try:
    params, cov = curve_fit(model, x, y, p0=[A0, B0, C0], maxfev=10000)
    A, B, C = params
    perr = np.sqrt(np.diag(cov))

    print(f"Fit-Parameter:")
    print(f"A = {A:.4e} ± {perr[0]:.4e}")
    print(f"B = {B:.4f} ± {perr[1]:.4f}")
    print(f"C = {C:.4f} ± {perr[2]:.4f}")

except RuntimeError as e:
    print(f"Fehler beim Fitten: {e}")
    exit()

plt.figure(figsize=(10, 6))

plt.plot(x, y, 'o', markersize=3, label="Daten")

# Plot der Fit-Kurve
plt.plot(x, model(x, *params), '-', label=fr"Fit: $A(\omega) = \frac{{{A:.2f}}}{{\sqrt{{({B:.2f}^2 - \omega^2)^2 + 4 \cdot {C:.2f}^2 \cdot \omega^2}}}}$")

# Achsenbeschriftungen und Titel
plt.xlabel("Anregungsfrequenz ($s^{-1}$)", fontsize=12)
plt.ylabel("Maximalamplitudeverhätnis", fontsize=12)

# Gitter und Legende
plt.grid(True)
plt.legend(fontsize=10)

# Layout anpassen und Plot anzeigen
plt.tight_layout()
plt.show()