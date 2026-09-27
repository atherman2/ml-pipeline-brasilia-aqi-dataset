from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

def criar_modelo_hiperparametros_otimizados(X0, y0):
    # O uso do TimeSeriesSplit previne o data leakage temporal durante a validação cruzada
    tscv = TimeSeriesSplit(n_splits=3)
    # O modelo principal (MLP) é instanciado sem pesos pré-treinados
    mlp_base = MLPClassifier(max_iter=1000, random_state=42)
    # Grid restrito a poucos parâmetros para otimizar o tempo de execução
    grid_parametros = {
        'hidden_layer_sizes': [(50,), (100,)],
        'learning_rate_init': [0.001, 0.01]
    }
    # A otimização ocorre utilizando estritamente os dados do conjunto histórico D0
    busca_otimizada = GridSearchCV(
        estimator=mlp_base,
        param_grid=grid_parametros,
        cv=tscv,
        scoring='f1',
        n_jobs=-1
    )
    busca_otimizada.fit(X0, y0)
    # O best_estimator_ já retorna o modelo treinado com todos os dados de X0 e y0 (M0)
    return busca_otimizada.best_estimator_, busca_otimizada.best_params_
