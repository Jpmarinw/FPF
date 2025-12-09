import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pmdarima as pm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import warnings

warnings.filterwarnings('ignore')

# 1. Carregar Dados
print("Carregando dados...")
df = pd.read_csv("../data/sensor_data.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)

# 2. Pré-processamento
print("Pré-processamento...")
# Reamostragem para 15min
df_resampled = df.resample('15min').mean()

# Preenchimento de Gaps (Interpolação)
df_resampled['potencia_watts'] = df_resampled['potencia_watts'].interpolate(method='linear')

# 3. EDA
print("Executando EDA...")
plt.figure(figsize=(15, 6))
plt.plot(df_resampled.index, df_resampled['potencia_watts'], label='Consumo de Energia')
plt.title('Consumo de Energia ao Longo do Tempo')
plt.legend()
plt.savefig('../images/eda_time_series.png')
plt.close()

# Teste de Estacionariedade
result = adfuller(df_resampled['potencia_watts'].dropna())
print(f'Estatística ADF: {result[0]}')
print(f'p-valor: {result[1]}')

# Decomposição
decomposition = seasonal_decompose(df_resampled['potencia_watts'].dropna(), period=96) # 96 * 15min = 24h
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.savefig('../images/eda_decomposition.png')
plt.close()

# Mapa de Calor
df_resampled['hour'] = df_resampled.index.hour
df_resampled['day_of_week'] = df_resampled.index.dayofweek
pivot_table = df_resampled.pivot_table(index='hour', columns='day_of_week', values='potencia_watts', aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='viridis')
plt.title('Consumo Médio de Energia: Hora vs Dia')
plt.savefig('../images/eda_heatmap.png')
plt.close()

# 4. Engenharia de Atributos
print("Engenharia de Atributos...")
# Atributos cíclicos
df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled['hour'] / 24)
df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled['hour'] / 24)
df_resampled['day_sin'] = np.sin(2 * np.pi * df_resampled['day_of_week'] / 7)
df_resampled['day_cos'] = np.cos(2 * np.pi * df_resampled['day_of_week'] / 7)

# Lags
df_resampled['lag_1'] = df_resampled['potencia_watts'].shift(1)
df_resampled['lag_2'] = df_resampled['potencia_watts'].shift(2)
df_resampled['lag_24h'] = df_resampled['potencia_watts'].shift(96) # 24h * 4 (chunks de 15min)

# Estatísticas móveis
df_resampled['rolling_mean'] = df_resampled['potencia_watts'].rolling(window=4).mean()
df_resampled['rolling_std'] = df_resampled['potencia_watts'].rolling(window=4).std()

df_model = df_resampled.dropna()

# 5. Modelagem
print("Modelagem...")

# Divisão Treino/Teste
train_size = int(len(df_model) * 0.8)
train, test = df_model.iloc[:train_size], df_model.iloc[train_size:]

# Baseline: AutoARIMA
print("Treinando AutoARIMA...")
try:
    # ARIMA simplificado para velocidade na demonstração
    model_arima = pm.auto_arima(train['potencia_watts'], seasonal=False, 
                                max_p=3, max_q=3, 
                                step_wise=True, suppress_warnings=True, error_action='ignore')
    preds_arima = model_arima.predict(n_periods=len(test))
    rmse_arima = np.sqrt(mean_squared_error(test['potencia_watts'], preds_arima))
    print(f"ARIMA RMSE: {rmse_arima}")
except Exception as e:
    print(f"Falha no ARIMA: {e}")
    rmse_arima = None

# XGBoost
print("Treinando XGBoost...")
features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'lag_1', 'lag_2', 'lag_24h', 'rolling_mean', 'rolling_std']
X_train, y_train = train[features], train['potencia_watts']
X_test, y_test = test[features], test['potencia_watts']

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)
preds_xgb = model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# 6. Conclusão
print("\n--- Resultados ---")
if rmse_arima:
    print(f"AutoARIMA RMSE: {rmse_arima:.4f}")
print(f"XGBoost RMSE: {rmse_xgb:.4f}")

with open("../output/results.txt", "w") as f:
    f.write("Comparação de Modelos:\n")
    if rmse_arima:
        f.write(f"AutoARIMA RMSE: {rmse_arima:.4f}\n")
    f.write(f"XGBoost RMSE: {rmse_xgb:.4f}\n")
