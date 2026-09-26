import numpy as np

class RegressaoLogisticaManual:
    def __init__(self, learning_rate=0.01, max_iter=1000):
        # Parâmetros essenciais claramente identificados
        self.learning_rate = learning_rate
        self.max_iter = max_iter # Critério de parada
        self.weights = None
        self.bias = None
        self.custo_historico = []

    def _sigmoid(self, z):
        # Tratamento mínimo de erro: previne overflow computacional no expoente
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Inicialização dos parâmetros
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Procedimento de treinamento
        for _ in range(self.max_iter):
            linear_model = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_model)

            # Função de custo (Cross-Entropy)
            epsilon = 1e-9 # Tratamento mínimo de erro: evita log(0)
            custo = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1 - y_pred + epsilon))
            self.custo_historico.append(custo)

            # Cálculo dos gradientes em formato matricial
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Atualização dos parâmetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_model)

    def predict(self, X, threshold=0.5):
        # Função de predição combinada à regra de decisão
        y_pred_prob = self.predict_proba(X)
        return (y_pred_prob > threshold).astype(int)
