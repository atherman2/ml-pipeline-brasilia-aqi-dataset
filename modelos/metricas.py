import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extrair_metricas_modelo(modelo, dados_treino, valores_alvo_treino, dados_aval, valores_alvo_aval):
    inicio = time.time()
    modelo.fit(dados_treino, valores_alvo_treino)
    previsoes = modelo.predict(dados_aval)
    tempo = time.time() - inicio
    return extrair_metricas_previsoes(valores_alvo_aval, previsoes, tempo)

def extrair_metricas_previsoes(valores_alvo, previsoes, tempo):
    return {
        "Tempo de execução": f"{tempo:.4f}",
        "Acurácia": f"{accuracy_score(valores_alvo, previsoes):.4f}",
        "Precisão": f"{precision_score(valores_alvo, previsoes, zero_division=0):.4f}",
        "Recall": f"{recall_score(valores_alvo, previsoes):.4f}",
        "F1-score": f"{f1_score(valores_alvo, previsoes, zero_division=0):.4f}",
        "Matriz de Confusão": f"{confusion_matrix(valores_alvo, previsoes)}"
    }
