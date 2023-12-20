import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./historico_eqm.csv')

plt.figure(figsize=(10, 5))  # Ajusta o tamanho do gráfico
plt.plot(df['Epoca'], df['Treinamento 4 EQM'], marker='o')  # Cria um gráfico de linha
plt.title('Época/Erro Treinamento 4')  # Adiciona um título ao gráfico
plt.xlabel('Época')  # Nomeia o eixo x
plt.ylabel('Erro')   # Nomeia o eixo y
plt.grid(True)  # Adiciona uma grade ao gráfico para facilitar a leitura
plt.show()  # Exibe o gráfico


plt.figure(figsize=(10, 5))  # Ajusta o tamanho do gráfico
plt.plot(df['Epoca'], df['Treinamento 3 EQM'], marker='o')  # Cria um gráfico de linha
plt.title('Época/Erro Treinamento 3')  # Adiciona um título ao gráfico
plt.xlabel('Época')  # Nomeia o eixo x
plt.ylabel('Erro')  # Nomeia o eixo y
plt.grid(True)  # Adiciona uma grade ao gráfico para facilitar a leitura
plt.show()  # Exibe o gráfico
