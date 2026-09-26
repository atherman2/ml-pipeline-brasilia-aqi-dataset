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

# %%
df_dataset_preparado = prepara_dataset()

# %%
periodos = pre_processamento_dataset(df_dataset_preparado)

# %%
