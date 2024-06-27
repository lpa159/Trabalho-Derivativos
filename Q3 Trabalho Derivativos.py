import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import random

# Parâmetros

r = 0.10  # Taxa de juros livre de risco contínua
mu = 0.15  # Taxa de retorno contínua esperada do underlying
q = 0.04  # Taxa de dividendos paga continuamente
T = 1  # Tempo até a maturidade em anos (250 dias úteis)
sigma = 0.40  # Volatilidade
S0 = 100  # Preço atual do underlying
n_steps = 250  # Número de passos (diário)
dt = T / n_steps  # Tamanho do passo
n_trajs = 10000  # Número de simulações
alpha = 0.05
c = 0.0025

# Parâmetros das opções
Kc = 110  # Strike price da call
Kp = 90   # Strike price da put

# Funções para cálculo do delta e preço do strangle considerando dividendos
def delta_call(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * norm.cdf(d1)

def delta_put(S, K, T, r, q, sigma):
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * (norm.cdf(d1) - 1)

def delta_short_strangle(S, K1, K2, T, r, q, sigma):
    return -delta_call(S, K2, T, r, q, sigma) - delta_put(S, K1, T, r, q, sigma)

def short_strangle_price(S, K1, K2, T, r, q, sigma):
    d1_call = (np.log(S / K2) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2_call = d1_call - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1_call) - K2 * np.exp(-r * T) * norm.cdf(d2_call)

    d1_put = (np.log(S / K1) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2_put = d1_put - sigma * np.sqrt(T)
    put_price = K1 * np.exp(-r * T) * norm.cdf(-d2_put) - S * np.exp(-q * T) * norm.cdf(-d1_put)

    return call_price + put_price

# Simulação das trajetórias do preço do ativo
price_paths = np.zeros((n_trajs, n_steps + 1))
price_paths[:, 0] = S0

for t in range(1, n_steps + 1):
    Z = np.random.normal(0, 1, n_trajs)
    price_paths[:, t] = price_paths[:, t - 1] * np.exp((mu - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

# Inicialização
strangle_init_price = short_strangle_price(S0, Kp, Kc, T, r, q, sigma)
deltas = np.zeros((n_trajs, n_steps + 1))
deltas[:, 0] = delta_short_strangle(S0, Kp, Kc, T, r, q, sigma)
cash_positions = np.zeros((n_trajs, n_steps + 1))
cash_positions[:, 0] = - deltas[:, 0] * S0

# Rebalanceamento dinâmico considerando dividendos
for t in range(1, n_steps + 1):
    T_t = T - t * dt
    deltas[:, t] = delta_short_strangle(price_paths[:, t], Kp, Kc, T_t, r, q, sigma)
    rebalance_indices = np.abs(deltas[:, t] - deltas[:, t - 1]) > np.abs(deltas[:, t - 1]) * alpha
    deltas[:, t] = np.where(rebalance_indices, deltas[:, t], deltas[:, t - 1])
    cash_positions[:, t] = (cash_positions[:, t - 1] * np.exp(r * dt) -
                            (deltas[:, t] - deltas[:, t - 1]) * price_paths[:, t] +
                            deltas[:, t - 1] * price_paths[:, t] * q * dt)

# Valor final do portfólio
portfolio_values = deltas * price_paths
final_portfolio_values = portfolio_values[:, -1] + cash_positions[:, -1]

# Calculando o payoff do short strangle
def short_strangle_payoff(ST, Kc, Kp):
    payoff_put = np.maximum(Kp - ST, 0)
    payoff_call = np.maximum(ST - Kc, 0)
    payoff = -payoff_call - payoff_put  # Mudança para refletir a venda das opções
    return payoff

ST = price_paths[:, -1]
VT = short_strangle_payoff(ST, Kc, Kp) + strangle_init_price * np.exp(r * T)

# Calculando o erro de hedging
epsilon_T = final_portfolio_values - VT

#Q2.3A

# Gráfico combinado: V_T e H_T por S_T
plt.figure(figsize=(12, 8))
plt.scatter(ST, VT, alpha=0.9, label='Lucro da Operação Estruturada (V_T)', color='blue', s=10)
plt.scatter(ST, final_portfolio_values, alpha=0.05, label='Valor da Estratégia Replicante (H_T)', color='red', s=10)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Preço do Underlying no Vencimento (ST)')
plt.ylabel('Valor')
plt.title('Lucro da Operação Estruturada vs. Valor da Estratégia Replicante no Vencimento')
plt.legend()
plt.grid(True)
plt.xlim(20, 200)
plt.ylim(-100, 100)

#Q2.3B

# Gráfico do erro de hedging ε_T por S_T
plt.figure(figsize=(14, 7))
plt.scatter(ST[epsilon_T >= 0], epsilon_T[epsilon_T >= 0], color='green', label='Erro de Hedging Positivo', s=10,
            alpha=0.5)
plt.scatter(ST[epsilon_T < 0], epsilon_T[epsilon_T < 0], color='red', label='Erro de Hedging Negativo', s=10, alpha=0.5)
plt.title('Erro de Hedging (ε_T) por Preço Final do Ativo Subjacente (S_T)')
plt.xlabel('Preço Final do Ativo Subjacente (S_T)')
plt.ylabel('Erro de Hedging (ε_T)')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True)
plt.xlim(20, 200)
plt.ylim(-50, 100)
plt.legend()


#Q2.3C

# Calculando os percentis do erro de hedging ε_T
percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
epsilon_percentiles = np.percentile(epsilon_T, percentiles)
percentile_table = pd.DataFrame({'Percentil': percentiles, 'Erro de Hedging': epsilon_percentiles})
print(percentile_table)

#Q2.3D

# Valores de alpha
alphas = [0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25]

# Função para realizar a simulação do delta hedging com custos de transação
def simulate_hedging_with_costs(alpha):
    price_paths = np.zeros((n_trajs, n_steps + 1))
    price_paths[:, 0] = S0

    for t in range(1, n_steps + 1):
        Z = np.random.normal(0, 1, n_trajs)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp((mu - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    strangle_init_price = short_strangle_price(S0, Kp, Kc, T, r, q, sigma)
    deltas = np.zeros((n_trajs, n_steps + 1))
    deltas[:, 0] = delta_short_strangle(S0, Kp, Kc, T, r, q, sigma)
    cash_positions = np.zeros((n_trajs, n_steps + 1))
    cash_positions[:, 0] = - deltas[:, 0] * S0
    transaction_costs = np.zeros((n_trajs, n_steps + 1))

    for t in range(1, n_steps + 1):
        T_t = T - t * dt
        deltas[:, t] = delta_short_strangle(price_paths[:, t], Kp, Kc, T_t, r, q, sigma)
        rebalance_indices = np.abs(deltas[:, t] - deltas[:, t - 1]) > np.abs(deltas[:, t - 1]) * alpha
        deltas[:, t] = np.where(rebalance_indices, deltas[:, t], deltas[:, t - 1])
        transaction_costs[:, t] = np.where(rebalance_indices, c * np.abs(deltas[:, t] - deltas[:, t - 1]) * price_paths[:, t], 0)
        cash_positions[:, t] = (cash_positions[:, t - 1] * np.exp(r * dt) -
                                (deltas[:, t] - deltas[:, t - 1]) * price_paths[:, t] +
                                deltas[:, t - 1] * price_paths[:, t] * q * dt -
                                transaction_costs[:, t])

    portfolio_values = deltas * price_paths
    final_portfolio_values = portfolio_values[:, -1] + cash_positions[:, -1]

    ST = price_paths[:, -1]
    VT = short_strangle_payoff(ST, Kc, Kp) + strangle_init_price * np.exp(r * T)
    epsilon_T = final_portfolio_values - VT

    return epsilon_T

# Simulação para diferentes valores de alpha
percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
results = {}

for alpha in alphas:
    epsilon_T = simulate_hedging_with_costs(alpha)
    epsilon_percentiles = np.percentile(epsilon_T, percentiles)
    results[alpha] = epsilon_percentiles

# Construção da tabela
percentile_table = pd.DataFrame(results, index=percentiles)
percentile_table.index.name = 'Percentil'
percentile_table.columns.name = 'Alpha'
print(percentile_table)

#Q2.3E

# Construir os gráficos
fig, ax = plt.subplots(figsize=(14, 8))

# Plotar os percentis para cada valor de alpha
for percentile in percentiles:
    ax.plot(alphas, percentile_table.loc[percentile], marker='o', label=f'{percentile}th Percentile')

ax.set_xlabel('Alpha')
ax.set_ylabel('Erro de Hedging (ε_T)')
ax.set_title('Percentis do Erro de Hedging (ε_T) em Função de Alpha')
ax.legend(title='Percentil')
ax.grid(True)
plt.show()

# Encontrar o maior valor do 25º percentil na tabela
max_25th_percentile_value = percentile_table.loc[25].max()
alpha_corresponding_to_max_25th = percentile_table.loc[25].idxmax()

print(f"O maior valor do 25º percentil é: {max_25th_percentile_value}")
print(f"Este valor corresponde ao alpha: {alpha_corresponding_to_max_25th}")

valor_cobrado =  abs(max_25th_percentile_value)
print("O valor cobrado deve ser:", valor_cobrado)



