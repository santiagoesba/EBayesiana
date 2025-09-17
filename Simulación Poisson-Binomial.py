import numpy as np
from scipy.stats import poisson, binom
import matplotlib.pyplot as plt

def comparar_binomial_poisson(lambda_param=12, p_fijo=0.5, N_sim=100000):
    """
    Compara la marginal de X (Poisson(lambda*p)), 
    la condicional X|N=n fijo (Binomial) y la simulación Monte Carlo.
    """
    # Rango de k suficientemente grande
    k = np.arange(0, 40)

    # Distribución marginal teórica: Poisson(lambda * p)
    px_marg = poisson.pmf(k, mu=lambda_param * p_fijo)

    # Distribución condicional X|N=fijo (por ejemplo N=20)
    n_fijo = 20
    px_cond = binom.pmf(k, n_fijo, p_fijo)

    # Simulación Monte Carlo
    N_simulados = np.random.poisson(lambda_param, size=N_sim)
    X_sim = np.random.binomial(N_simulados, p_fijo)
    px_sim = np.bincount(X_sim, minlength=len(k))[:len(k)] / N_sim

    # Graficar
    plt.bar(k - 0.25, px_marg, width=0.25, label='Marginal X ~ Poi(lambda*p)', color='black')
    plt.bar(k, px_cond, width=0.25, label=f'Condicional X|N={n_fijo}', color = 'white')
    plt.bar(k + 0.25, px_sim, width=0.25, alpha=0.7, label='Simulación Monte Carlo', color='pink')
    plt.xlabel('X')
    plt.xlim(0, 12)
    plt.ylabel('Probabilidad')
    plt.title(f'Comparación: λ={lambda_param}, p={p_fijo}')
    plt.legend()
    plt.show()

    return k, px_marg, px_cond, px_sim

k, px_marg, px_cond, px_sim = comparar_binomial_poisson(lambda_param=12, p_fijo=0.3)
