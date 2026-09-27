import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def metricas_treinamento_predicao(modelo, dados_treino, valores_alvo_treino, dados_aval, valores_alvo_aval):
    inicio_treinamento = time.perf_counter()
    modelo.fit(dados_treino, valores_alvo_treino)
    tempo_treinamento = time.perf_counter() - inicio_treinamento
    return metricas_predicao(modelo, tempo_treinamento, dados_aval, valores_alvo_aval)

def metricas_predicao(modelo, tempo_treinamento, dados, valores_alvo):
    inicio_predicao = time.perf_counter()
    predicoes = modelo.predict(dados)
    tempo_predicao = time.perf_counter() - inicio_predicao
    return extrair_metricas_de_predicoes(valores_alvo, predicoes, tempo_treinamento, tempo_predicao)

def extrair_metricas_de_predicoes(valores_alvo, predicoes, tempo_treinamento, tempo_predicao):
    return {
        "Tempo de treinamento": f"{tempo_treinamento:.4f}",
        "Tempo de predição": f"{tempo_predicao:.4f}",
        "Acurácia": f"{accuracy_score(valores_alvo, predicoes):.4f}",
        "Precisão": f"{precision_score(valores_alvo, predicoes, zero_division=0):.4f}",
        "Recall": f"{recall_score(valores_alvo, predicoes):.4f}",
        "F1-score": f"{f1_score(valores_alvo, predicoes, zero_division=0):.4f}",
        "Matriz de Confusão": f"{confusion_matrix(valores_alvo, predicoes)}"
    }
