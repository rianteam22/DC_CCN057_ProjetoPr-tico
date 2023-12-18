# DC_CCN057 Projeto Prático
Este repositório contém implementações em Python baseadas no livro `Redes Neurais Artificiais Para Engenharia E Ciências Aplicadas. Curso Prático`

Projeto prático 1 (aproximação de funções)
Capitulo 5 - Página 164

# Implementação de Perceptron Multicamadas (MLP)



Este código fornece uma implementação de um Perceptron Multicamadas simples, um tipo de rede neural artificial. Abaixo está uma explicação detalhada da classe e seus métodos:



## Classe MultilayerPerceptron



- `__init__(self, tamanho_entrada, tamanho_oculto, tamanho_saida, taxa_aprendizado=0.1, epsilon=0.000001)`: Construtor que inicializa os pesos para as camadas oculta e de saída com valores aleatórios. Também define a taxa de aprendizado e o épsilon, que é usado para o critério de parada baseado no erro quadrático médio (EQM).



- `logistica(self, x)`: A função de ativação logística (sigmoide). Mapeia qualquer entrada 'x' para um valor entre 0 e 1, o que é útil para problemas de classificação binária.



- `logistica_derivada(self, x)`: A derivada da função logística. Isso é usado durante a etapa de retropropagação para calcular os gradientes.



- `forward(self, x)`: Função de passagem para frente que recebe uma entrada 'x' e a passa pela rede, retornando a saída. Esta função aplica os pesos às entradas, usa a função de ativação e, em seguida, propaga os dados para a próxima camada.



- `backward(self, x, d)`: Função de passagem para trás que propaga o erro de volta pela rede para atualizar os pesos. É aqui que a aprendizagem real ocorre usando o algoritmo de descida do gradiente.



- `calcular_erro_individual(self, desejado, saida)`: Uma função para calcular o erro de uma única amostra de dados. Isso é metade do quadrado da diferença entre a saída desejada 'd' e a saída real 'saida'.



- `calcular_eqm(self, X, D)`: Função para calcular o erro quadrático médio (EQM) para um conjunto de amostras 'X' com suas saídas desejadas correspondentes 'D'. Isso fornece uma medida de quão bem a rede está se desempenhando.



- `treinamento(self, X, D)`: A função de treinamento que recebe um conjunto de entradas 'X' e saídas desejadas 'D' para treinar a rede. Os pesos são atualizados até que a mudança no EQM entre as épocas seja menor ou igual a épsilon.



- `predict(self, X)`: Após a rede ter sido treinada, esta função pode ser usada para fazer previsões sobre novos dados.



## Uso



Para usar esta implementação, instancie a classe `MultilayerPerceptron` com os tamanhos desejados para as camadas de entrada, oculta e de saída, juntamente com os parâmetros de aprendizagem. Em seguida, chame o método `treinamento` com seus dados de treinamento para treinar o modelo. Após o treinamento, use o método `predict` para fazer previsões.


