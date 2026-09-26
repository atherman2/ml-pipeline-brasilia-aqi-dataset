import pandas as pd
from sklearn.preprocessing import StandardScaler

COL_DATA = "Date"
COL_AQI_FUTURO = "Future_24h_AQI"
COL_AVISO_AQI = "AQI_24h_Warning"
SALTO = 2
DATA_INICIO = "2024-01-01 00:00:00"
DATA_FIM = "2025-12-31 23:59:59"
COLUNAS_NECESSARIAS = ["Date", "CO", "NO2", "SO2", "O3", "PM2.5", "PM10", "AQI", COL_AQI_FUTURO]
COLUNAS_REMOVER = ["City", "source_file_path"]
COLUNAS_NUMERICAS = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10", "AQI"]
VARIAVEIS_EXPLICATIVAS = ["CO", "NO2", "SO2", "O3", "PM2.5", "PM10", "AQI"]
LIMIAR_AQI = 25.0

def pre_processamento_dataset(df):
    df = ordenacao_temporal(df)
    df = criar_campo_aqi_futuro(df)
    df = recorte_tempo(df, COL_DATA, DATA_INICIO, DATA_FIM)
    df = downsampling(df, salto=SALTO)
    df = remocao_colunas(df)
    df = tratamento_ausentes(df)
    df = remocao_duplicatas(df)
    df = limiarizar_coluna_aqi(df)
    d0, d1, d2 = split_temporal(df)
    d0, d1, d2 = normalizar_dados(d0, d1, d2)
    return separacao_variavel_alvo(d0, d1, d2)

def downsampling(df, salto=1):
    """
    Filtra o dataset temporalmente e aplica técnicas de amostragem.
    """
    # Amostragem (Redução de linhas)
    if salto > 1:
        # Pega 1 a cada 'salto' linhas
        df = df.iloc[::salto]
    return df

def recorte_tempo(df, col_data, data_inicio, data_fim):
    # Recorte da Janela Temporal
    mask = (df[col_data] >= data_inicio) & (df[col_data] <= data_fim)
    df = df[mask].copy()
    return df

def tratamento_ausentes(df):
    df = df.dropna(subset=COLUNAS_NECESSARIAS).copy()
    return df

def remocao_colunas(df):
    for coluna in COLUNAS_REMOVER:
        if coluna in df.columns:
            df.drop(columns=[coluna], inplace=True)
    return df

def remocao_duplicatas(df):
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=[COL_DATA])
    return df

def criar_campo_aqi_futuro(df):
    df_futuro = df[["Date", "AQI"]].copy()
    df_futuro.rename(columns={"AQI": "Future_24h_AQI"}, inplace=True)
    df_futuro["Date"] = df_futuro["Date"] - pd.Timedelta(hours=24)
    df_final = pd.merge(df, df_futuro, on="Date", how="inner")
    return df_final

def ordenacao_temporal(df):
    # Garante que a coluna é datetime
    if not pd.api.types.is_datetime64_any_dtype(df[COL_DATA]):
        df[COL_DATA] = pd.to_datetime(df[COL_DATA])
    # Ordenação cronológica estrita (Obrigatório na especificação)
    df = df.sort_values(by=COL_DATA).reset_index(drop=True)
    return df

def limiarizar_coluna_aqi(df):
    df[COL_AVISO_AQI] = (df[COL_AQI_FUTURO] > LIMIAR_AQI).astype(int)
    df.drop(columns=COL_AQI_FUTURO, inplace=True)
    return df

def split_temporal(df):
    n_total = len(df)
    n_d0 = int(n_total * 0.60)
    n_d1 = int(n_total * 0.20)

    # Cortes cronológicos (60% / 20% / 20%)
    D0 = df.iloc[:n_d0].copy()
    D1 = df.iloc[n_d0 : n_d0 + n_d1].copy()
    D2 = df.iloc[n_d0 + n_d1:].copy()

    return D0, D1, D2

def normalizar_dados(d0, d1, d2):
    """
    Aplica a normalização (Z-score) garantindo que não haja vazamento de dados.
    """
    scaler = StandardScaler()
    # Ajuste e transformação exclusivos em D0
    d0[COLUNAS_NUMERICAS] = scaler.fit_transform(d0[COLUNAS_NUMERICAS])
    # Apenas transformação em D1 e D2 usando o mapeamento aprendido em D0
    d1[COLUNAS_NUMERICAS] = scaler.transform(d1[COLUNAS_NUMERICAS])
    d2[COLUNAS_NUMERICAS] = scaler.transform(d2[COLUNAS_NUMERICAS])

    return d0, d1, d2

def separacao_variavel_alvo(d0, d1, d2):
    coluna_alvo = COL_AVISO_AQI
    periodos = []
    for d in [d0, d1, d2]:
        periodos.append((
            d[VARIAVEIS_EXPLICATIVAS].values,
            d[coluna_alvo].values
        ))
    return periodos
