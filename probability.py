## @package probability
#  Pakiet funkcji probabilistycznych.
#
#  Szczegółowy opis pakietu.

import math
import numpy as np
import matplotlib.pyplot as plt

## @brief Klasa reprezentująca rozkład normalny.
#
# Klasa umożliwia obliczanie wartości funkcji gęstości (PDF) i dystrybuanty (CDF) 
# dla rozkładu normalnego oraz rysowanie ich wykresów.
# 
# Funkcja gęstości (PDF) jest opisana wzorem:
# \f[ f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x-\mu)^2}{2\sigma^2}} \f]
# Dystrybuanta (CDF) jest opisana wzorem:
# \f[ F(x) = \frac{1}{2} \left[1 + \text{erf}\left(\frac{x-\mu}{\sqrt{2}\sigma}\right)\right] \f]
class NormalDistribution:
    ## Konstruktor.
    #
    # @param mu Wartość oczekiwana rozkładu.
    # @param sigma Odchylenie standardowe rozkładu.
    def __init__(self, mu, sigma):
        if sigma <= 0:
            raise ValueError("Odchylenie standardowe (sigma) musi być większe od zera.")
        ## Średnia (wartość oczekiwana) rozkładu normalnego.
        self.mu = mu
        ## Odchylenie standardowe rozkładu normalnego.
        self.sigma = sigma

    ## @brief Oblicza wartość funkcji gęstości (PDF) w punkcie x.
    # 
    # @param x Punkt, dla którego obliczana jest wartość funkcji gęstości.
    # 
    # @return Wartość funkcji PDF dla x.
    def pdf(self, x):
        return (1 / (math.sqrt(2 * math.pi) * self.sigma)) * math.exp(-((x - self.mu)**2) / (2 * self.sigma**2))

    ## @brief Oblicza wartość funkcji dystrybuanty (CDF) w punkcie x.
    #
    # @param x Punkt, dla którego obliczana jest wartość dystrybuanty.
    #
    # @return Wartość funkcji CDF dla x.
    def cdf(self, x):
        return 0.5 * (1 + math.erf((x - self.mu) / (math.sqrt(2) * self.sigma)))

    ## @brief Rysuje wykresy funkcji PDF i CDF w zadanym przedziale.
    #
    # @image html normal_pdf_cdf.png width=800px
    #
    # @param x_min Dolna granica przedziału.
    # @param x_max Górna granica przedziału.
    # @param num_points Liczba punktów na osi x (domyślnie 1000).
    def plot_pdf_cdf(self, x_min, x_max, num_points=1000):
        x = np.linspace(x_min, x_max, num_points)
        pdf_values = [self.pdf(xi) for xi in x]
        cdf_values = [self.cdf(xi) for xi in x]

        plt.figure(figsize=(12, 6))

        # Wykres PDF
        plt.subplot(1, 2, 1)
        plt.plot(x, pdf_values, label='PDF')
        plt.title('Funkcja gęstości (PDF)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.grid(True)
        plt.legend()

        # Wykres CDF
        plt.subplot(1, 2, 2)
        plt.plot(x, cdf_values, label='CDF', color='orange')
        plt.title('Dystrybuanta (CDF)')
        plt.xlabel('x')
        plt.ylabel('F(x)')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    nd = NormalDistribution(mu=0, sigma=1)
    print("PDF(0):", nd.pdf(0))
    print("CDF(0):", nd.cdf(0))
    nd.plot_pdf_cdf(-5, 5)