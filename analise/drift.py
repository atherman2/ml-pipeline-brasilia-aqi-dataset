import pandas as pd
from scipy.stats import ks_2samp

def analisar_drift_ks(d0, d1, d2, colunas_features):
    resultados = []

    for col in colunas_features:
        stat_d1, p_valor_d1 = ks_2samp(d0[col], d1[col])
        stat_d2, p_valor_d2 = ks_2samp(d0[col], d2[col])

        resultados.append({
            "Atributo": col,
            "Drift D0->D1": p_valor_d1 < 0.05,
            "P-valor D1": p_valor_d1,
            "KS Stat D1": stat_d1,
            "Drift D0->D2": p_valor_d2 < 0.05,
            "P-valor D2": p_valor_d2,
            "KS Stat D2": stat_d2
        })

    df_resultados = pd.DataFrame(resultados)
    return df_resultados
