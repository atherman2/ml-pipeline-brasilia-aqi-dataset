# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
from tratamento_dataset.preparacao import prepara_dataset
df_dataset_preparado = prepara_dataset()

# %%
from tratamento_dataset.pre_processamento import pre_processamento_dataset
periodos = pre_processamento_dataset(df_dataset_preparado)

# %%
from sklearn.linear_model import LogisticRegression
from modelos.regressao_logistica import RegressaoLogisticaManual
from modelos.metricas import metricas_treinamento_predicao
print("Métricas do modelo de regressão logística implementado:")
print(metricas_treinamento_predicao(
    RegressaoLogisticaManual(),
    periodos[0][0], periodos[0][1], periodos[1][0], periodos[1][1]
))
print()
print("Métricas do modelo de regressão logística da biblioteca scikit-learn:")
print(metricas_treinamento_predicao(
    LogisticRegression(),
    periodos[0][0], periodos[0][1], periodos[1][0], periodos[1][1]
))

# %%
from sklearn.neural_network import MLPClassifier
from modelos.mlp import criar_modelo_hiperparametros_otimizados
from modelos.metricas import metricas_predicao
from util.seeds import GerenciadorSeeds
import copy
import numpy as np
import time
import json

inicio_otimizacao = time.perf_counter()
_, hiperparametros_m0 = criar_modelo_hiperparametros_otimizados(periodos[0][0], periodos[0][1])
print(f"Otimização concluída em {time.perf_counter() - inicio_otimizacao:.2f}s")
print(f"Arquitetura definida para os experimentos: {hiperparametros_m0}\n")

gseeds = GerenciadorSeeds()
SEED_BASE = 42
NUM_ITERACOES = 16

modelos_nomes = ["M0_D1", "M0_D2", "MFT_D2", "MRT_D2", "MREC_D2"]
resultados = {modelo: {} for modelo in modelos_nomes}

def registrar_metricas(nome_modelo, dict_metricas):
    for metrica, valor in dict_metricas.items():
        if metrica not in resultados[nome_modelo]:
            resultados[nome_modelo][metrica] = {"valores": []}

        if metrica == "Matriz de Confusão":
            resultados[nome_modelo][metrica]["valores"].append(valor)
        else:
            resultados[nome_modelo][metrica]["valores"].append(float(valor))

for i in range(NUM_ITERACOES):
    seed = gseeds.get_seed_i(i, SEED_BASE)
    print(f"Executando iteração {i + 1}/{NUM_ITERACOES} (Seed: {seed})...")

    m0 = MLPClassifier(**hiperparametros_m0, random_state=seed)
    inicio = time.perf_counter()
    m0.fit(periodos[0][0], periodos[0][1])
    tempo_m0 = time.perf_counter() - inicio

    registrar_metricas("M0_D1", metricas_predicao(m0, tempo_m0, periodos[1][0], periodos[1][1]))
    registrar_metricas("M0_D2", metricas_predicao(m0, tempo_m0, periodos[2][0], periodos[2][1]))

    mft = copy.deepcopy(m0)
    inicio = time.perf_counter()
    mft.partial_fit(periodos[1][0], periodos[1][1])
    tempo_ft = time.perf_counter() - inicio
    metricas_mft = metricas_predicao(mft, tempo_m0 + tempo_ft, periodos[2][0], periodos[2][1])
    metricas_mft["Tempo de treinamento fine-tuning"] = tempo_ft
    registrar_metricas("MFT_D2", metricas_mft)

    drt_X = np.concatenate([periodos[0][0], periodos[1][0]], axis=0)
    drt_y = np.concatenate([periodos[0][1], periodos[1][1]], axis=0)
    mrt = MLPClassifier(**hiperparametros_m0, random_state=seed)
    inicio = time.perf_counter()
    mrt.fit(drt_X, drt_y)
    tempo_mrt = time.perf_counter() - inicio
    registrar_metricas("MRT_D2", metricas_predicao(mrt, tempo_mrt, periodos[2][0], periodos[2][1]))

    mrec = MLPClassifier(**hiperparametros_m0, random_state=seed)
    inicio = time.perf_counter()
    mrec.fit(periodos[1][0], periodos[1][1])
    tempo_mrec = time.perf_counter() - inicio
    registrar_metricas("MREC_D2", metricas_predicao(mrec, tempo_mrec, periodos[2][0], periodos[2][1]))

for modelo, metricas in resultados.items():
    for metrica, dados in metricas.items():
        if metrica != "Matriz de Confusão":
            dados["media"] = np.mean(dados["valores"])
            dados["desvio_padrao"] = np.std(dados["valores"])

print("Resultados do Experimento:")
print(json.dumps(resultados, indent=4, ensure_ascii=False))


# %%
