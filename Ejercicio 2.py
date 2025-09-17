import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
# En 1996, Gary Kasparov se enfrentó a la computadora de IBM Deep Blue en 6 juegos de ajedrez. De los 6 juegos, Kasparov ganó 3 de ellos, empató 2 y perdió uno. En 1997 se enfrentarían de nuevo en otros 6 juegos de ajedrez. Sea θla probabilidad de que Kasparov le gane en un juego de ajedrez a Deep Blue. Suponga que los juegos son independientes entre sí y que Kasparov tiene la misma probabilidad en cada uno de ellos. Considerando los resultados de 1996 construimos la siguiente distribución previa sobre θ:
    #θ 0.2 0.5 0.8
    #p(θ) 0.1 0.25 0.65
    #i. Grafique la distribución predictiva previa, es decir p(y)
    #ii. Sea Y el número de juegos que Kasparov podría ganar en 1997. Con la distribución previa del inciso anterior, calcule la probabilidad de que Kasparov gane 0,1,...,6 de 10 los siguientes juegos. Es decir, calcule la distribución predictiva previa y grafíquela. ¿Cuál es el número más probable de juegos que ganaría Kasparov?
    #iii. En la competencia de 1997, Kasparov únicamente ganó uno de los 6 juegos de ajedrez contra Deep Blue, es decir Y = 1. Con esta información, calcule la distribución posterior de θ y grafíquela.
    #iv. Compare la distribución previa y posterior de θ, y las distribuciones predictivas previa y posterior para Y. Comente sus resultados
# --- Datos del problema ---
theta_vals = np.array([0.2, 0.5, 0.8])   
prior_probs = np.array([0.1, 0.25, 0.65])  

# --- i. Distribución predictiva previa ---
p_y1 = np.sum(theta_vals * prior_probs)  # probabilidad de ganar
p_y0 = np.sum((1 - theta_vals) * prior_probs)  # probabilidad de no ganar (complementario)
vector_probs = np.array([p_y0, p_y1])

df = pd.DataFrame({
    "y": [0,1],
    "predictiva": vector_probs
})
print(df)

# Grafica
plt.bar(["Kasparov no gana", "Kasparov gana"], vector_probs,
        color=["blue", "green"])
plt.title("Distribución predictiva previa p(y)")
plt.ylabel("Probabilidad")
plt.show()


# --- ii. Distribución de Y ~ número de juegos ganados, en 6 partidas ---
n = 6
k_vals = np.arange(0, n+1)

# Binomiales
predictive_Y = np.zeros_like(k_vals, dtype=float)
for theta, w in zip(theta_vals, prior_probs):
    predictive_Y += w * binom.pmf(k_vals, n, theta)

# Graficas
plt.bar(k_vals, predictive_Y, color="skyblue", edgecolor="black")
plt.title("Distribución predictiva previa de Y")
plt.xlabel("Número de juegos ganados")
plt.ylabel("Probabilidad")
plt.show()

# Número más probable de juegos que ganaría Kasparov
most_probable = k_vals[np.argmax(predictive_Y)]
df2 = pd.DataFrame({
    "Y": [0,1,2,3,4,5,6],
    "predictiva": predictive_Y
})
print(df2)


print("Número más probable de juegos ganados:", most_probable)


# In[ ]:




