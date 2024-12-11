## @package discrete_math
#  Pakiet funkcji matematyki dyskretnej.
#
#  Szczegółowy opis pakietu.

import math

## @brief Oblicza n-ty wyraz ciągu Fibonacciego.
#
# Funkcja wyznacza n-ty wyraz ciągu Fibonacciego, wykorzystując formułę Bineta:
# \f[ F(n) = \frac{\varphi^n - (-\varphi)^{-n}}{\sqrt{5}} \f]
# gdzie @f$\varphi@f$ to złoty podział:
# \f[ \varphi = \frac{1 + \sqrt{5}}{2} \f]
#
# @param n Indeks wyrazu ciągu Fibonacciego (\f$n \geq 0\f$).
#
# @return n-ty wyraz ciągu Fibonacciego jako liczba całkowita.
#
# @note Ze względu na ograniczenia precyzji liczb zmiennoprzecinkowych,
# wyniki mogą nie być dokładne dla bardzo dużych wartości \f$ n \f$.
def fibonacci(n):
    if n < 0:
        raise ValueError("Indeks ciągu Fibonacciego nie może być ujemny.")

    sqrt_5 = math.sqrt(5)
    phi = (1 + sqrt_5) / 2
    psi = (1 - sqrt_5) / 2

    # Formuła Bineta
    result = (phi**n - psi**n) / sqrt_5

    return round(result)
