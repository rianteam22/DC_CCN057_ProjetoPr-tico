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

    def logistica(self, x, B=1):
        # Função de ativação logística
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
        historico_eqm = []  # Lista para armazenar o histórico do EQM
        cont = 0
        while True:
            eqm_anterior = self.calcular_eqm(X, D)
            historico_eqm.append(eqm_anterior)  # Armazena o EQM da época atual
        
            for x, d in zip(X, D):
             self.forward(x)
             self.backward(x, d)
        
            eqm_atual = self.calcular_eqm(X, D)
            self.epocas += 1
        
            if abs(eqm_atual - eqm_anterior) <= self.Epsilon:
                cont += 1
                if cont >= 2:
                    print(f"Convergência alcançada após {self.epocas} épocas com EQM de {eqm_atual}.")
                    break
        return historico_eqm  # Retorna o histórico do EQM após o treinamento

    def operacao(self, X):
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

# Separação dos dados em inputs (x) e outputs (d)
x = pdDataFrame.iloc[:, :3].values  # Seleciona as três primeiras colunas para input
d = pdDataFrame.iloc[:, 3].values  # Seleciona a quarta coluna para output

x_teste = pdDataFrameTeste.iloc[:, :3].values

input_size = 3  # Número de entradas
hidden_size = 10 # Número de neurônios na camada oculta
output_size = 1 # Número de saídas

historicos_eqm = []

for i in range(5):
    mlp = MultilayerPerceptron(input_size, hidden_size, output_size)
    historico_eqm = mlp.treinamento(x, d)  # Treinamento e armazenamento do histórico de EQM
    historicos_eqm.append(historico_eqm)  # Adiciona o histórico de EQM à lista
    mlp.operacao(x_teste)

# Criação de um DataFrame para armazenar todos os históricos de EQM
df_historicos_eqm = pd.DataFrame(historicos_eqm).T  # Transposta para ter as épocas nas linhas
df_historicos_eqm.columns = [f'Treinamento {i+1} EQM' for i in range(5)]  # Nomeando as colunas

# Salvar o DataFrame em um arquivo CSV
df_historicos_eqm.to_csv('historico_eqm.csv', index_label='Epoca')