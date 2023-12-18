import numpy as np
import pandas as pd

class MultilayerPerceptron:
    def __init__(self, tamanho_entrada, tamanho_oculto, tamanho_saida, taxa_aprendizado=0.1, epsilon=0.000001):
        # Inicializa os pesos das camadas oculta e de saída com valores aleatórios
        self.w_oculta = np.random.rand(tamanho_entrada + 1, tamanho_oculto)  # Incluindo o bias
        self.w_saida = np.random.rand(tamanho_oculto + 1, tamanho_saida)  # Incluindo o bias
        self.N = taxa_aprendizado
        self.Epsilon = epsilon
        self.epocas = 0
        # Impressões removidas para brevidade

    def logistica(self, x, B=1):
        # Função de ativação logística (sigmoidal)
        return 1 / (1 + np.exp(-B * x))

    def logistica_derivada(self, x):
        # Derivada da função de ativação logística
        logis = self.logistica(x)
        return logis * (1 - logis)

    def forward(self, x):
        # Propaga a entrada x através da rede e retorna a saída
        x = np.insert(x, 0, -1)  # Adiciona o bias
        self.entrada_oculta = np.dot(x, self.w_oculta)
        self.oculta = self.logistica(self.entrada_oculta)
        self.oculta = np.insert(self.oculta, 0, -1)  # Adiciona o bias
        pot_oculta = np.dot(self.oculta, self.w_saida)
        self.saida = self.logistica(pot_oculta)
        return self.saida

    def backward(self, x, d):
        # Retropropaga o erro e atualiza os pesos
        erro_saida = d - self.saida
        gradiente_saida = erro_saida * self.logistica_derivada(self.saida)
        self.w_saida += self.N * np.outer(self.oculta, gradiente_saida)
        erro_oculta = gradiente_saida.dot(self.w_saida[1:].T)
        gradiente_oculta = erro_oculta * self.logistica_derivada(self.entrada_oculta)
        self.w_oculta += self.N * np.outer(np.insert(x, 0, -1), gradiente_oculta)

    def calcular_erro_individual(self, desejado, saida):
        # Calcula o erro para uma única amostra
        return 0.5 * np.sum((desejado - saida) ** 2)

    def calcular_eqm(self, X, D):
        # Calcula o erro quadrático médio para o conjunto de dados
        eqm_total = 0
        for x, d in zip(X, D):
            saida = self.forward(x)
            eqm_total += self.calcular_erro_individual(d, saida)
        return eqm_total / len(X)

    def treinamento(self, X, D):
        cont = 0
        while True:
            eqm_anterior = self.calcular_eqm(X, D)
#            print(f"EQM anterior: {eqm_anterior}")
            for x, d in zip(X, D):
                self.forward(x)
                self.backward(x, d)
            
            eqm_atual = self.calcular_eqm(X, D)
#            print(f"EQM atual: {eqm_atual}")
#            print(f"Época: {self.epocas}")
            self.epocas += 1
            
            if abs(eqm_atual - eqm_anterior) <= self.Epsilon:
                cont += 1
                if cont >= 2:
                    print("Erro quadrático médio suficientemente pequeno alcançado por 2 épocas consecutivas"
                    + f" com valor absoluto de {abs(eqm_atual - eqm_anterior)}")
                    print(f"Convergência alcançada após {self.epocas} épocas com EQM de {eqm_atual}.")
                    break

    def predict(self, X):
        # Realiza predições com base nos dados de entrada
        predicoes = []
        for x in X:
            predicoes.append(self.forward(x))
        print(np.array(predicoes))
        return np.array(predicoes)

# Caminho do arquivo Excel
arquivo_xls_treino = "./Treinamento_PMC_Aproximação_Universal_PPA.xlsx"
arquivo_xls_teste = "./Teste_PMC_Aproximação_Universal_PPA.xlsx"

# Leitura do arquivo Excel para um DataFrame
pdDataFrame = pd.read_excel(arquivo_xls_treino)
pdDataFrameTeste = pd.read_excel(arquivo_xls_teste)
print(pdDataFrame)

# Verifica se o DataFrame tem 4 colunas (3 para input, 1 para output)
# Separação dos dados em inputs (x) e outputs (d)
x = pdDataFrame.iloc[:, :3].values  # Seleciona as três primeiras colunas para input
d = pdDataFrame.iloc[:, 3].values  # Seleciona a quarta coluna para output

x_teste = pdDataFrameTeste.iloc[:, :3].values

input_size = 3  # Número de entradas
hidden_size = 10 # Número de neurônios na camada oculta
output_size = 1 # Número de saídas

for i in range(5):
    mlp = MultilayerPerceptron(input_size, hidden_size, output_size)
    mlp.treinamento(x, d)
    mlp.predict(x_teste)