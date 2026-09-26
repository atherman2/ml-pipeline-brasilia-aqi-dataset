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
from tratamento_dataset.pre_processamento import pre_processamento_dataset
from sklearn.linear_model import LogisticRegression
from modelos.regressao_logistica import RegressaoLogisticaManual
from modelos.metricas import extrair_metricas_modelo, extrair_metricas_previsoes

# %%
df_dataset_preparado = prepara_dataset()

# %%
periodos = pre_processamento_dataset(df_dataset_preparado)

# %%
print("Métricas do modelo de regressão logística implementado:")
print(extrair_metricas_modelo(
    RegressaoLogisticaManual(),
    periodos[0][0], periodos[0][1], periodos[1][0], periodos[1][1]
))
print()
print("Métricas do modelo de regressão logística da biblioteca scikit-learn")
print(extrair_metricas_modelo(
    LogisticRegression(),
    periodos[0][0], periodos[0][1], periodos[1][0], periodos[1][1]
))

# %%
