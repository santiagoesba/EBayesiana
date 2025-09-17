import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta, binom
# Dos personas están interesadas en estimar la proporción p de estudiantes que usan bicicleta para llegar a la universidad. Ana usa la siguiente previa basándose en una encuesta vieja que encontró:
#p 0.1 0.2 0.3 0.4 0.5
#g(p) 0.5 0.2 0.2 0.05 0.05
#Bart decide usar como previa Beta(3,12).
#i. Grafique ambas previas, obtenga ambos valores esperados, desviaciones estándar y modas ¿podría decir que Ana y Bart tienen ideas similares?
#ii. La universidad comunica a Bart y Ana que el siguiente semestre entrarán n = 12,50,100 alumnos. Obtenga la distribición predictiva de Ana y Bart para la proporción de alumnos que usarán bicicleta para llegar a la universidad.
# --------------------------
# Previa de Ana
# --------------------------
p_vals = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
g_p = np.array([0.5, 0.2, 0.2, 0.05, 0.05])  # probabilidades

# media
mean_ana = np.sum(p_vals * g_p)
# varianza
var_ana = np.sum((p_vals - mean_ana)**2 * g_p)
std_ana = np.sqrt(var_ana)
# moda (valor más probable)
mode_ana = p_vals[np.argmax(g_p)]

# --------------------------
# Previa de Bart (Beta(3,12))
# --------------------------
a, b = 3, 12
mean_bart = a / (a + b)
var_bart = a * b / ((a + b)**2 * (a + b + 1))
std_bart = np.sqrt(var_bart)
# moda de la Beta (cuando a,b > 1)
mode_bart = (a - 1) / (a + b - 2)

# --------------------------
# Graficar previas
# --------------------------
x = np.linspace(0, 1, 200)
plt.figure(figsize=(10,5))
# Ana (discreta)
plt.stem(p_vals, g_p, linefmt='b-', markerfmt='bo', basefmt=" ", label="Ana (discreta)")
# Bart (beta)
plt.plot(x, beta.pdf(x, a, b), 'r-', lw=2, label="Bart Beta(3,12)")
plt.title("Previas de Ana y Bart")
plt.xlabel("p")
plt.ylabel("Densidad / Probabilidad")
plt.legend()
plt.grid()
plt.show()

print("Resultados:")
print(f"Ana -> media={mean_ana:.3f}, std={std_ana:.3f}, moda={mode_ana}")
print(f"Bart -> media={mean_bart:.3f}, std={std_bart:.3f}, moda={mode_bart:.3f}")

# --------------------------
# ii) Distribuciones predictivas
# --------------------------
def predictiva_ana(n):
    # mezcla de binomiales
    k = np.arange(0, n+1)
    pmf = np.zeros_like(k, dtype=float)
    for pi, wi in zip(p_vals, g_p):
        pmf += wi * binom.pmf(k, n, pi)
    return k, pmf

def predictiva_bart(n):
    # distribución predictiva Beta-Binomial
    k = np.arange(0, n+1)
    pmf = binom.pmf(k, n, mean_bart)  # approx (real es Beta-Binomial)
    # Mejor: usar fórmula Beta-Binomial:
    from scipy.special import comb, beta as B
    pmf = comb(n, k) * B(k+a, n-k+b) / B(a,b)
    return k, pmf

# Graficar predictivas para distintos n
for n in [12, 50, 100]:
    k_ana, pmf_ana = predictiva_ana(n)
    k_bart, pmf_bart = predictiva_bart(n)
    
    plt.figure(figsize=(10,5))
    plt.plot(k_ana/n, pmf_ana, 'bo-', label="Ana predictiva")
    plt.plot(k_bart/n, pmf_bart, 'r-', label="Bart predictiva")
    plt.title(f"Distribución predictiva (n={n})")
    plt.xlabel("Proporción de alumnos en bicicleta")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.grid()
    plt.show()
