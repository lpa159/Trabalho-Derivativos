import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parâmetros
r = 0.10
mu = 0.15
q = 0.04
T = 1.0
sigma = 0.40
S0 = 100
S_min_t = 20
S_max_t = 200
delta_S = 0.1
Kp = 90
Kc = 110
times = [0, 0.25 * T, 0.50 * T, 0.75 * T, T]

# Fórmula de Black-Scholes para opções europeias
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# Fórmula de Black-Scholes para delta de opções europeias
def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    if option_type == "call":
        delta = norm.cdf(d1)
    elif option_type == "put":
        delta = norm.cdf(d1) - 1
    return delta

# Valor da strangle vendida em diferentes momentos e preços
S_values = np.arange(S_min_t, S_max_t, delta_S)
V_t = np.zeros((len(times), len(S_values)))
Delta_t = np.zeros((len(times), len(S_values)))

for j, t in enumerate(times):
    for i, S in enumerate(S_values):
        put_price = black_scholes(S, Kp, T-t, r, sigma, "put")
        call_price = black_scholes(S, Kc, T-t, r, sigma, "call")
        V_t[j, i] = -(put_price + call_price)  # strangle vendida é vender ambas as opções

        put_delta = black_scholes_delta(S, Kp, T-t, r, sigma, "put")
        call_delta = black_scholes_delta(S, Kc, T-t, r, sigma, "call")
        Delta_t[j, i] = -(put_delta + call_delta)  # delta da strangle vendida é a soma dos deltas das opções vendidas

# Plotar o gráfico de Valor de Mercado
plt.figure(figsize=(14, 8))
for j, t in enumerate(times):
    plt.plot(S_values, V_t[j, :], label=f't = {t:.2f}T')

plt.xlabel('Preço do Subjacente $S_t$')
plt.ylabel('Valor de Mercado $V_t$')
plt.title('Valor de Mercado da Operação Short Strangle ao Longo do Tempo')
plt.legend()
plt.grid(True)


# Plotar o gráfico de Delta
plt.figure(figsize=(14, 8))
for j, t in enumerate(times):
    plt.plot(S_values, Delta_t[j, :], label=f't = {t:.2f}T')

plt.xlabel('Preço do Subjacente $S_t$')
plt.ylabel('Delta $\Delta_t$')
plt.title('Delta da Operação Short Strangle ao Longo do Tempo')
plt.legend()
plt.grid(True)


# Plotar gráfico 3D para Valor de Mercado
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
T_mesh, S_mesh = np.meshgrid(times, S_values)

for j in range(len(times)):
    ax.plot(S_values, np.full_like(S_values, times[j]), V_t[j, :])

ax.set_xlabel('Preço do Subjacente $S_t$')
ax.set_ylabel('Tempo $t$')
ax.set_zlabel('Valor de Mercado $V_t$')
ax.set_title('Valor de Mercado da Strangle Vendida ao Longo do Tempo')


# Plotar gráfico 3D para Delta
fig = plt.figure(figsize=(14, 8))
ax = fig.add_subplot(111, projection='3d')
for j in range(len(times)):
    ax.plot(S_values, np.full_like(S_values, times[j]), Delta_t[j, :])

ax.set_xlabel('Preço do Subjacente $S_t$')
ax.set_ylabel('Tempo $t$')
ax.set_zlabel('Delta $\Delta_t$')
ax.set_title('Delta da Strangle Vendida ao Longo do Tempo')
plt.show()