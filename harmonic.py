## @package harmonic
#  Równanie ruchu harmonicznego z tłumieniem wiskotycznym
#
# Rozważmy równanie różniczkowe drugiego rzędu:
# \f[ m \frac{\text{d}^2x}{\text{d}t^2} + c \frac{\text{d}x}{\text{d}t} + kx = 0 \f]
# Możemy przekształcić je do układu równań różniczkowych pierwszego rzędu wprowadzając zmienne stanu:
# \f[ \mathbf{z} = \begin{bmatrix} x \\ \dot{x} \end{bmatrix}, \f]
# co daje:
# \f[ \frac{\text{d}\mathbf{z}}{\text{d}t} = \begin{bmatrix} \dot{x} \\ \frac{1}{m}(-c\dot{x} - kx) \end{bmatrix} \f]
# i prowadzi do postaci macierzowej:
# \f[ \frac{\text{d}\mathbf{z}}{\text{d}t} = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix} \mathbf{z} \f]
# Macierz układu \f$ A \f$ pozwala na znalezienie rozwiązania zarówno metodami numerycznymi, jak i analitycznymi:
# \f[ A = \begin{bmatrix} 0 & 1 \\ -k/m & -c/m \end{bmatrix} \f]
# Jeżeli dodatkowo zadane jest wymuszenie \f$ F(t) \f$, to ruch definiowany jest za pomocą równania różniczkowego:
# \f[ m \frac{\text{d}^2x}{\text{d}t^2} + c \frac{\text{d}x}{\text{d}t} + kx = F(t), \f]
# które w formie macierzowej ma postać:
# \f[ \frac{\text{d}\mathbf{z}}{\text{d}t} = A \mathbf{z} + \begin{bmatrix} 0 \\ F(t) \end{bmatrix}. \f]

# Import biblioteki numerycznej
import numpy as np
# Import biblioteki do rozwiązywania równań różniczkowych zwyczajnych
from scipy.integrate import solve_ivp
# Import biblioteki do rysowania wykresów
import matplotlib.pyplot as plt

## @brief Funkcja symulująca ruch harmoniczny.
#
# @image html harmonic.png width=1200px
def harmonic():
    # Parametry układu
    m = 1.0     # masa
    c = 0.5     # tłumienie
    k = 100.0     # sztywność
    ni = 1.591 # częstotliwość wymuszeia

    # Równanie ruchu
    # m·x" + c·x' + k·x = sin(2π·ν·t)

    # Macież różniczki równania
    A = np.array([
        [0, 1],
        [-k/m, -c/m]
    ])

    # Obliczanie wartości własnych (eigenvalues) i wektorów własnych (eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eig(A)

    # Wyświetlanie wartości własnych
    print("Wartości własne macierzy A:")
    print(eigenvalues)

    # Konwersja częstotliwości do Hz
    frequencies_hz = np.abs(eigenvalues.imag) / (2 * np.pi)
    print("\nCzęstotliwości własne (Hz):")
    print(frequencies_hz)

    # Funkcja definiująca równania różniczkowe
    def ode_system(t, y):
        # Wyraz wolny równania pochodnej
        b = np.array([0, np.sin(2*np.pi*(ni)*t)])
        # Wektor pochodnej w punkcie (t, y) wyznaczony jako A*y+b
        return np.dot(A, y) + b

    # Warunki początkowe
    y0 = np.array([0, 0])

    # Przedział czasu
    t_span = (0, 50)  # od 0 do 50 sekund
    t_eval = np.linspace(0, 50, 5000)  # punkty czasowe do oceny

    # Rozwiązanie ODE
    sol = solve_ivp(ode_system, t_span, y0, t_eval=t_eval)

    # Wykres wyników
    plt.figure(figsize=(18.0,3.0))
    plt.plot(sol.t, sol.y[0], label='Przemieszczenie (x)')
    plt.plot(sol.t, sol.y[1], label='Prędkość (v)')
    plt.xlabel('Czas [s]')
    plt.ylabel('Wartość')
    plt.legend()
    plt.title('Ruch sprężyny zamocowanej do ściany')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    harmonic()
