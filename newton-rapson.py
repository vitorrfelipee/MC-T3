import numpy as np
import matplotlib.pyplot as plt
import struct

# Constantes e funções utilizadas no cálculo
sqrt2 = 1.4142135623730950488016887242097

def sqrt_2e(e):
    return 2.0**(e // 2) * (sqrt2 if e % 2 else 1.0)

def log_base_2(x):
    return int(np.log2(x))

def calcula_e_f(x):
    e = log_base_2(x)
    f = x / (2**e) - 1
    return e, f

def raiz_calculada(x):
    e, f = calcula_e_f(x)
    return sqrt_2e(e) * (1 + f / 2)

P_64 = 2**61
Q_64 = 2**52

def calcular_chute_inicial_float_union(A):
    if A == 0.0:
        return 0.0

    val = FloatUnion(A)
    val.x -= Q_64
    val.x >>= 1
    val.x += P_64

    return struct.unpack('d', struct.pack('Q', val.x))[0]

class FloatUnion:
    def __init__(self, f):
        self.f = f
        self.x = struct.unpack('Q', struct.pack('d', f))[0]

def newton_raphson_sqrt(A, x0, precisao):
    x = x0
    iter_count = 0
    while abs(x**2 - A) > precisao:
        x = 0.5 * (x + A / x)
        iter_count += 1
    return x, iter_count

def calcular_chute_inicial_numpy(A):
    return np.sqrt(A)

# Preparação dos dados
A_values = np.arange(0.05, 20, 0.05)
precisao = 1e-14
iteracoes_calculados = []
iteracoes_newton = []
iteracoes_numpy = []

for A in A_values:
    valor_raiz_t1 = raiz_calculada(A)
    _, iter_calculado = newton_raphson_sqrt(A, valor_raiz_t1, precisao)
    iteracoes_calculados.append(iter_calculado)
    
    valor_raiz_newton = calcular_chute_inicial_float_union(A)
    _, iter_newton = newton_raphson_sqrt(A, valor_raiz_newton, precisao)
    iteracoes_newton.append(iter_newton)
    
    valor_raiz_numpy = calcular_chute_inicial_numpy(A)
    _, iter_numpy = newton_raphson_sqrt(A, valor_raiz_numpy, precisao)
    iteracoes_numpy.append(iter_numpy)

# Plotando os gráficos
plt.figure(figsize=(14, 7))

# Método Calculado
plt.subplot(1, 2, 1)
plt.plot(A_values, iteracoes_calculados, label='Método Calculado', color='blue')
plt.xlabel('Valor de A')
plt.ylabel('Número de Iterações')
plt.title('Iterações: Método Calculado')
plt.grid(True)
plt.legend()

# Método Float Union
plt.subplot(1, 2, 2)
plt.plot(A_values, iteracoes_newton, label='Método Float Union', color='red')
plt.xlabel('Valor de A')
plt.ylabel('Número de Iterações')
plt.title('Iterações: Método Float Union')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Plotando os gráficos de iterações dos dois métodos no mesmo gráfico para comparação direta
plt.figure(figsize=(12, 6))
plt.plot(A_values, iteracoes_calculados, label='Método Calculado', color='blue')
plt.plot(A_values, iteracoes_newton, label='Método Float Union', color='red')
plt.plot(A_values, iteracoes_numpy, label='Método Numpy', color='green')
plt.xlabel('Valor de A')
plt.ylabel('Número de Iterações')
plt.title('Comparação de Iterações: Métodos Calculado vs. Float Union')
plt.legend()
plt.grid(True)
plt.show()

# Calculando as médias de iterações
media_iteracoes_calculados = np.mean(iteracoes_calculados)
media_iteracoes_newton = np.mean(iteracoes_newton)
media_iteracoes_numpy = np.mean(iteracoes_numpy)
(media_iteracoes_calculados, media_iteracoes_newton, media_iteracoes_numpy)
print("Média de iterações para o método calculado:", media_iteracoes_calculados)
print("Média de iterações para o método Float Union:", media_iteracoes_newton)
print("Média de iterações para o método Numpy:", media_iteracoes_numpy)



