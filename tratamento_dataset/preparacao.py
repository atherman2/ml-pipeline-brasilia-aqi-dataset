import pandas as pd

ARQUIVOS = [
    "youssefelebiary/air-quality-2024/versions/1/Brasilia_Air_Quality.csv",
    "youssefelebiary/global-air-quality-2025-6-cities/versions/1/Brasilia_Air_Quality.csv"
]
COL_DATA = "Date"
COLUNAS_DROP = [
    "Unnamed: 0",
    "CO2"
]

def prepara_dataset():
    df = concatena_df_arquivos()
    df = prepara_coluna_data(df)
    return df

def concatena_df_arquivos():
    lista_df_anos = []
    for arq in ARQUIVOS:
        df = pd.read_csv(arq)
        df["source_file_path"] = arq
        for coluna in COLUNAS_DROP:
            if coluna in df.columns:
                df.drop(columns=[coluna], inplace=True)
        lista_df_anos.append(df)
    df = pd.concat(lista_df_anos, ignore_index=True)
    return df

def prepara_coluna_data(df):
    df[COL_DATA] = pd.to_datetime(df[COL_DATA])
    return df
