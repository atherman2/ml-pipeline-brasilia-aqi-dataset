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
import copy
import numpy as np
import time

inicio = time.perf_counter()
m0, hiperparametros_m0 = criar_modelo_hiperparametros_otimizados(periodos[0][0], periodos[0][1])
tempo_m0 = time.perf_counter() - inicio
metricas = metricas_predicao(m0, tempo_m0, periodos[1][0], periodos[1][1])
print("Métricas de M0 sobre D1:")
print(metricas)
print()
metricas = metricas_predicao(m0, tempo_m0, periodos[2][0], periodos[2][1])
print("Métricas de M0 sobre D2:")
print(metricas)
print()

mft = copy.deepcopy(m0)
inicio = time.perf_counter()
mft.partial_fit(periodos[1][0], periodos[1][1])
tempo_ft = time.perf_counter() - inicio
metricas = metricas_predicao(mft, tempo_m0 + tempo_ft, periodos[2][0], periodos[2][1])
metricas["Tempo de treinamento fine-tuning"] = tempo_ft
print("Métricas de MFT sobre D2:")
print(metricas)
print()

drt = (
    np.concatenate([periodos[0][0], periodos[1][0]], axis=0),
    np.concatenate([periodos[0][1], periodos[1][1]], axis=0),
)
mrt = MLPClassifier(**hiperparametros_m0, random_state=42)
inicio = time.perf_counter()
mrt.fit(drt[0], drt[1])
tempo_mrt = time.perf_counter() - inicio
metricas = metricas_predicao(mrt, tempo_mrt, periodos[2][0], periodos[2][1])
print("Métricas de MRT sobre D2:")
print(metricas)
print()

mrec = MLPClassifier(**hiperparametros_m0, random_state=42)
inicio = time.perf_counter()
mrec.fit(periodos[1][0], periodos[1][1])
tempo_mrec = time.perf_counter() - inicio
metricas = metricas_predicao(mrec, tempo_mrec, periodos[2][0], periodos[2][1])
print("Métricas de MREC sobre D2:")
print(metricas)
print()

# %%
