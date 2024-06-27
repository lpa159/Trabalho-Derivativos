import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd

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

# Parâmetros das opções
Kc = 110  # Strike price da call
Kp = 90   # Strike price da put

#Payoff da Operação Short Strangle

S_values = np.linspace(20, 200, 1000)
payoff = -np.maximum(Kp - S_values, 0) - np.maximum(S_values - Kc, 0)

plt.figure(figsize=(14, 8))
plt.plot(S_values, payoff, lw=2, color='blue')
plt.xlabel('Preço do Subjacente no Vencimento (ST)')
plt.ylabel('Payoff')
plt.title('Payoff da Operação de Short Strangle')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.grid(True)


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

# Gráfico das trajetórias simuladas
plt.figure(figsize=(14, 7))
for i in range(n_trajs):
    plt.plot(price_paths[i], alpha=0.5)
plt.title('Trajetórias Simuladas do Preço do Ativo Subjacente')
plt.xlabel('Passos de Tempo')
plt.ylabel('Preço do Ativo Subjacente (S)')
plt.grid(True)

#Q2.2A

# Gráfico combinado: V_T e H_T por S_T
plt.figure(figsize=(12, 8))
plt.scatter(ST, VT, alpha=0.9, label='Lucro da Operação Estruturada (V_T)', color='blue', s=10)
plt.scatter(ST, final_portfolio_values, alpha=0.05, label='Valor da Estratégia Replicante (H_T)', color='red', s=10)
plt.axhline(0, color='black', linewidth=0.5)
plt.xlabel('Preço do Underlying no Vencimento (ST)')
plt.ylabel('Valor')
plt.title('Payoff da Operação Estruturada vs. Valor da Estratégia Replicante no Vencimento')
plt.legend()
plt.grid(True)
plt.xlim(20, 200)
plt.ylim(-100, 100)

#Q2.2B

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



#Q2.2C

# Calculando os percentis do erro de hedging ε_T
percentiles = [0, 5, 10, 25, 50, 75, 90, 95, 100]
epsilon_percentiles = np.percentile(epsilon_T, percentiles)
percentile_table = pd.DataFrame({'Percentil': percentiles, 'Erro de Hedging': epsilon_percentiles})
print(percentile_table)

#Q2.2D
# Parâmetros adicionais para a questão D
n_steps_values = [10, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]
c = 0.0025  # Custo percentual (25 bps)

# Função para realizar a simulação do delta hedging com custos de transação
def simulate_hedging_with_costs(n_steps):
    dt = T / n_steps
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
        transaction_costs[:, t] = c * np.abs(deltas[:, t] - deltas[:, t - 1]) * price_paths[:, t]
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

# Construir a tabela de percentis do erro de hedging
percentiles_df = pd.DataFrame(index=n_steps_values, columns=percentiles)

for n in n_steps_values:
    epsilon_T = simulate_hedging_with_costs(n)
    percentiles_df.loc[n] = np.percentile(epsilon_T, percentiles)

# Exibir a tabela
print(percentiles_df)

#Q2.2E

import matplotlib.pyplot as plt

# Reutilizando a tabela de percentis do erro de hedging obtida anteriormente

# Plotando os gráficos
plt.figure(figsize=(14, 8))
for percentile in percentiles:
    plt.plot(percentiles_df.index, percentiles_df[percentile], label=f'{percentile}th Percentile')

plt.xlabel('Número de Passos (n_steps)')
plt.ylabel('Erro de Hedging (ε_T)')
plt.title('Percentis do Erro de Hedging para Diferentes Valores de n_steps')
plt.legend()
plt.grid(True)
#plt.show()

#Q2.2F Analisar o tradeoff entre custo e o erro de hedging

# Encontrar o maior valor do 25º percentil na tabela
max_25_percentile_value = percentiles_df[25].max()
steps_corresponding_max_25 = percentiles_df[percentiles_df[25] == max_25_percentile_value].index[0]


print(f"O maior valor do 25º percentil é: {max_25_percentile_value}")
print(f"Este valor corresponde ao step: {steps_corresponding_max_25}")

valor_cobrado =  abs(max_25_percentile_value)
print("O valor cobrado deve ser:", valor_cobrado)

