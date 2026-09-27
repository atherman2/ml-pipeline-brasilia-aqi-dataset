import pandas as pd
from sklearn.inspection import permutation_importance

def explicabilidade_permutacao(modelo, X, y, nomes_features, n_repeats=10, random_state=42):
    resultado = permutation_importance(
        estimator=modelo,
        X=X,
        y=y,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    importancias = []

    for i in resultado.importances_mean.argsort()[::-1]:
        importancias.append({
            "Atributo": nomes_features[i],
            "Importancia_Media": resultado.importances_mean[i],
            "Desvio_Padrao": resultado.importances_std[i]
        })

    df_importancias = pd.DataFrame(importancias)
    return df_importancias
