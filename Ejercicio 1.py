import math
import pandas as pd
import matplotlib.pyplot as plt
# Los cucos son aves parasitarias de puesta, esto significa que depositan sus huevos en los nidos de otras aves (el huésped). Para que sea el huésped la que alimente a sus crías. Lisa es una ornitóloga que estudia la tasa de éxito, π, de que la cría de un cuco sobreviva al menos una semana. En un estudio previo, otro investigador sugiere la siguiente distribución para π: π 0.6 0.65 0.7 0.75. p(π) 0.3 0.4 0.2 0.1 Lisa recolecta sus propios datos, en los que observa que de 15 crías de cucos, 10 sobrevivieron la primer semana. Calcule la distribución posterior de π.

# Valores sugeridos de pi y probabilidades a priori
pi_values = [0.6, 0.65, 0.7, 0.75]
prior = [0.3, 0.4, 0.2, 0.1]

# Datos observados
n = 15   # número total de crías
k = 10   # número de crías que sobrevivieron

# Función de probabilidad binomial
def binom_pmf(k, n, p):
    return math.comb(n, k) * (p**k) * ((1-p)**(n-k))

# Calculamos la verosimilitud (Probabilidad de los datos dado que el parametro fuera pi) -> Lo aplicaremos en F.Bayes
likelihoods = [binom_pmf(k, n, p) for p in pi_values]

# Calculamos el numerador
numerador = [prior[i] * likelihoods[i] for i in range(len(pi_values))]

# Calculamos el denominador, llamamos evidencia a P(datos)
evidence = sum(numerador)

# Calculamos probabilidad posterior 
posterior = [u / evidence for u in numerador]

# Creamos una tabla con los resultados (Distribucion a posteriori)
df = pd.DataFrame({
    "pi": pi_values,
    "posteriori": posterior
})
print(df)


# === Gráfico de la distribución posterior ===
plt.bar([str(p) for p in pi_values], posterior, color="skyblue", edgecolor="black")
plt.xlabel("Valores de π")
plt.ylabel("Probabilidad posterior")
plt.title("Distribución posterior de π")
plt.show()
