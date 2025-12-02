# Bibliotecas
from skforecast.ForecasterBaseline import ForecasterEquivalentDate
from skforecast.ForecasterSarimax import ForecasterSarimax
from skforecast.ForecasterAutoregMultiVariate import ForecasterAutoregMultiVariate
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.Sarimax import Sarimax
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, QuantileRegressor
from sklearn.svm import LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, BaggingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.preprocessing import PowerTransformer
import pandas as pd
import numpy as np
import os


# Definições e configurações globais
h = 12 # horizonte de previsão
inicio_treino = pd.to_datetime("2004-01-01") # amostra inicial de treinamento
semente = 1984 # semente para reprodução


# Função para transformar dados, conforme definido nos metadados
def transformar(x, tipo):

  switch = {
      "1": lambda x: x,
      "2": lambda x: x.diff(),
      "3": lambda x: x.diff().diff(),
      "4": lambda x: np.log(x),
      "5": lambda x: np.log(x).diff(),
      "6": lambda x: np.log(x).diff().diff()
  }

  if tipo not in switch:
      raise ValueError("Tipo inválido")

  return switch[tipo](x)


# Planilha de metadados
metadados = (
    pd.read_excel(
        io = "https://docs.google.com/spreadsheets/d/1x8Ugm7jVO7XeNoxiaFPTPm1mfVc3JUNvvVqVjCioYmE/export?format=xlsx",
        sheet_name = "Metadados",
        dtype = str,
        index_col = "Identificador"
        )
    .filter(["Transformação"])
)

# Importa dados online
dados_brutos_m = pd.read_parquet("dados/df_mensal.parquet")
dados_brutos_t = pd.read_parquet("dados/df_trimestral.parquet")
dados_brutos_a = pd.read_parquet("dados/df_anual.parquet")

# Converte frequência
dados_tratados = (
    dados_brutos_m
    .asfreq("MS")
    .join(
        other = dados_brutos_a.asfreq("MS").ffill(),
        how = "outer"
        )
    .join(
        other = (
            dados_brutos_t
            .filter(["us_gdp", "pib"])
            .dropna()
            .assign(us_gdp = lambda x: ((x.us_gdp.rolling(4).mean() / x.us_gdp.rolling(4).mean().shift(4)) - 1) * 100)
            .asfreq("MS")
            .ffill()
        ),
        how = "outer"
    )
    .rename_axis("data", axis = "index")
)

# Separa Y
y = dados_tratados.endividamento_familias_exhabit.dropna()
y.head()

# Separa X
x = dados_tratados.drop(labels = ["endividamento_familias_exhabit"], axis = "columns").copy()
x.head()

# Computa transformações - Corrigido Gemini
# Get a list of columns in x that also exist in metadados index
columns_to_process = [col for col in x.columns if col in metadados.index and col not in ["saldo_caged_antigo", "saldo_caged_novo"]]

for col in columns_to_process:
  x[col] = transformar(x[col], metadados.loc[col, "Transformação"])

x.head()

# Filtra amostra
y = y[y.index >= inicio_treino]
x_alem_de_y = x.query("index >= @y.index.max()")
x = x.loc[y.index] # Align x to y's index

# Conta por coluna proporção de NAs em relação ao nº de obs. de Y
prop_na = x.isnull().sum() / y.shape[0]
prop_na.sort_values(ascending = False)

# Remove variáveis que possuem mais de 20% de NAs
x = x.drop(labels = prop_na[prop_na >= 0.2].index.to_list(), axis = "columns")
x.head()

# Preenche NAs restantes com a vizinhança
x = x.bfill().ffill()
x

# Seleção final de variáveis
x_reg = [
    "expec_selic",
    "igv_r",
    "us_gov_sec_3m",
    "prod_ind_papel"
    ]
# + 1 lags
x_reg

# Reestima melhor modelo com amostra completa
forecaster = ForecasterAutoreg(
    regressor = Ridge(),
    lags = 1,
    transformer_y = PowerTransformer(),
    transformer_exog = PowerTransformer()
    )
forecaster.fit(y, x[x_reg])
forecaster

# Período de previsão fora da amostra
periodo_previsao = pd.date_range(
    start = forecaster.last_window.index[0] + pd.offsets.QuarterBegin(1),
    end = forecaster.last_window.index[0] + pd.offsets.QuarterBegin(h + 1),
    freq = "QS"
    )
periodo_previsao

# Coleta dados de expectativas da Selic
dados_focus_selic = (
    pd.read_csv(
        filepath_or_buffer = f"https://olinda.bcb.gov.br/olinda/servico/Expectativas/versao/v1/odata/ExpectativasMercadoTop5Selic?$filter=Data%20ge%20'{forecaster.last_window.index[0].strftime('%Y-%m-%d')}'%20and%20tipoCalculo%20eq%20'C'&$format=text/csv",
        decimal = ",",
        converters = {
            "Data": pd.to_datetime,
            "DataReferencia": lambda x: pd.to_datetime(x, format = "%m/%Y")
            }
        ))
dados_focus_selic.head()

# Constrói cenário para expectativas de juros (swaps_di_360)
dados_cenario_swaps_di_360 = (
    dados_focus_selic
    .query("Data == Data.max()")
    .rename(columns = {"mediana": "swaps_di_360"})
    .head(12)
    .filter(["swaps_di_360"])
    .set_index(periodo_previsao)
)
dados_cenario_swaps_di_360

# Constrói cenário para Índice de Valores de Garantia de Imóveis Residenciais Financiados (igv_r)
dados_cenario_igv_r = (
    x
    .filter(["igv_r"])
    .dropna()
    .query("index >= '2015-01-01'")
    .assign(trim = lambda x: x.index.quarter)
    .groupby(["trim"], as_index = False)
    .igv_r
    .median()
    .set_index("trim")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(trim = lambda x: x.data.dt.quarter)
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("trim")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)
dados_cenario_igv_r

# Constrói cenário para Market Yield on U.S. Treasury Securities at 3-Month Constant Maturity (us_gov_sec_3m)
dados_cenario_us_gov_sec_3m = (
    x
    .filter(["us_gov_sec_3m"])
    .dropna()
    .query("index >= '2015-01-01'")
    .assign(trim = lambda x: x.index.quarter)
    .groupby(["trim"], as_index = False)
    .us_gov_sec_3m
    .median()
    .set_index("trim")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(trim = lambda x: x.data.dt.quarter)
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("trim")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)
dados_cenario_us_gov_sec_3m

# Constrói cenário para Fabricação de celulose, papel e produtos de papel (prod_ind_papel)
dados_cenario_prod_ind_papel = (
    x
    .filter(["prod_ind_papel"])
    .dropna()
    .query("index >= '2020-01-01'")
    .assign(trim = lambda x: x.index.quarter)
    .groupby(["trim"], as_index = False)
    .prod_ind_papel
    .median()
    .set_index("trim")
    .join(
        other = (
            periodo_previsao
            .rename("data")
            .to_frame()
            .assign(trim = lambda x: x.data.dt.quarter)
            .drop("data", axis = "columns")
            .reset_index()
            .set_index("trim")
        ),
        how = "outer"
    )
    .sort_values(by = "data")
    .set_index("data")
)
dados_cenario_prod_ind_papel

# Junta cenários (apenas variáveis em x_reg)
dados_cenarios = (
    dados_cenario_swaps_di_360
    .join(
        other = [
            dados_cenario_igv_r,
            dados_cenario_us_gov_sec_3m,
            dados_cenario_prod_ind_papel
            ],
        how = "outer"
        )
    .asfreq("QS")
)
dados_cenarios

# Produz previsões
previsao = forecaster.predict_interval(
    steps = h,
    exog = dados_cenarios,
    n_boot = 5000,
    random_state = semente
    )
previsao

# Salvar previsões
pasta = "previsao"
if not os.path.exists(pasta):
  os.makedirs(pasta)
  
pd.concat(
    [y.rename("ndividamento das famílias exceto crédito habitacional (RNDBF)"),
     previsao.pred.rename("Previsão"),
     previsao.lower_bound.rename("Intervalo Inferior"),
     previsao.upper_bound.rename("Intervalo Superior"),
    ],
    axis = "columns"
    ).to_parquet("previsao/endivfam.parquet")
