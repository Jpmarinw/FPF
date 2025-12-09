import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.cells.append(nbf.v4.new_markdown_cell("""# Desafio de Previsão Industrial - Pipeline Ponta a Ponta

Este notebook implementa um pipeline completo de Machine Learning para previsão de consumo de energia.

## 1. Configuração e Importações
"""))

nb.cells.append(nbf.v4.new_code_cell("""import pandas as pd
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
plt.style.use('seaborn-v0_8')
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 2. Carregamento dos Dados
Carregando os dados sintéticos dos sensores.
"""))

nb.cells.append(nbf.v4.new_code_cell("""df = pd.read_csv("../data/sensor_data.csv", parse_dates=['timestamp'])
df.set_index('timestamp', inplace=True)
print(df.head())
print(df.info())
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 3. Pré-processamento
- Reamostragem para 15 minutos.
- Preenchimento de lacunas (gaps) usando interpolação linear.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Reamostragem para 15min
df_resampled = df.resample('15min').mean()

# Preenchimento de Gaps (Interpolação)
df_resampled['potencia_watts'] = df_resampled['potencia_watts'].interpolate(method='linear')

print(f"Novo formato após reamostragem: {df_resampled.shape}")
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 4. Análise Exploratória de Dados (EDA)
Visualizando a série temporal, verificando estacionariedade e decompondo componentes.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Gráfico da Série Temporal
plt.figure(figsize=(15, 6))
plt.plot(df_resampled.index, df_resampled['potencia_watts'], label='Consumo de Energia')
plt.title('Consumo de Energia ao Longo do Tempo')
plt.legend()
plt.show()

# Teste de Estacionariedade (ADF)
result = adfuller(df_resampled['potencia_watts'].dropna())
print(f'Estatística ADF: {result[0]}')
print(f'p-valor: {result[1]}')
if result[1] < 0.05:
    print("A série é estacionária.")
else:
    print("A série NÃO é estacionária.")
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Decomposição
decomposition = seasonal_decompose(df_resampled['potencia_watts'].dropna(), period=96) # 96 * 15min = 24h
fig = decomposition.plot()
fig.set_size_inches(15, 10)
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Mapa de Calor (Dia vs Hora)
df_resampled['hour'] = df_resampled.index.hour
df_resampled['day_of_week'] = df_resampled.index.dayofweek
pivot_table = df_resampled.pivot_table(index='hour', columns='day_of_week', values='potencia_watts', aggfunc='mean')

plt.figure(figsize=(10, 8))
sns.heatmap(pivot_table, cmap='viridis', annot=False)
plt.title('Consumo Médio de Energia: Hora vs Dia')
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 5. Engenharia de Atributos (Feature Engineering)
Criando atributos cíclicos para tempo, defasagens (lags) e estatísticas móveis.
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Atributos Cíclicos
df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled['hour'] / 24)
df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled['hour'] / 24)
df_resampled['day_sin'] = np.sin(2 * np.pi * df_resampled['day_of_week'] / 7)
df_resampled['day_cos'] = np.cos(2 * np.pi * df_resampled['day_of_week'] / 7)

# Lags (Defasagens)
df_resampled['lag_1'] = df_resampled['potencia_watts'].shift(1)
df_resampled['lag_2'] = df_resampled['potencia_watts'].shift(2)
df_resampled['lag_24h'] = df_resampled['potencia_watts'].shift(96) # 24h * 4 (chunks de 15min)

# Estatísticas Móveis (Rolling stats)
df_resampled['rolling_mean'] = df_resampled['potencia_watts'].rolling(window=4).mean()
df_resampled['rolling_std'] = df_resampled['potencia_watts'].rolling(window=4).std()

df_model = df_resampled.dropna()
print(df_model.head())
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 6. Modelagem
Comparando AutoARIMA (Estatístico) e XGBoost (Machine Learning).
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Divisão Treino/Teste
train_size = int(len(df_model) * 0.8)
train, test = df_model.iloc[:train_size], df_model.iloc[train_size:]
print(f"Tamanho Treino: {len(train)}, Tamanho Teste: {len(test)}")
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Modelo 1: AutoARIMA
# Nota: ARIMA Sazonal pode ser lento. Usando parâmetros simplificados para demonstração.
print("Treinando AutoARIMA...")
model_arima = pm.auto_arima(train['potencia_watts'], seasonal=False, 
                            max_p=3, max_q=3, 
                            step_wise=True, suppress_warnings=True, error_action='ignore')

preds_arima = model_arima.predict(n_periods=len(test))
rmse_arima = np.sqrt(mean_squared_error(test['potencia_watts'], preds_arima))
print(f"ARIMA RMSE: {rmse_arima}")

# Plotar resultados ARIMA
plt.figure(figsize=(15, 5))
plt.plot(test.index, test['potencia_watts'], label='Real')
plt.plot(test.index, preds_arima, label='Previsão ARIMA', alpha=0.7)
plt.legend()
plt.title('Previsão ARIMA vs Real')
plt.show()
"""))

nb.cells.append(nbf.v4.new_code_cell("""# Modelo 2: XGBoost
print("Treinando XGBoost...")
features = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'lag_1', 'lag_2', 'lag_24h', 'rolling_mean', 'rolling_std']
X_train, y_train = train[features], train['potencia_watts']
X_test, y_test = test[features], test['potencia_watts']

model_xgb = XGBRegressor(n_estimators=100, learning_rate=0.1)
model_xgb.fit(X_train, y_train)
preds_xgb = model_xgb.predict(X_test)
rmse_xgb = np.sqrt(mean_squared_error(y_test, preds_xgb))
print(f"XGBoost RMSE: {rmse_xgb}")

# Plotar resultados XGBoost
plt.figure(figsize=(15, 5))
plt.plot(test.index, y_test, label='Real')
plt.plot(test.index, preds_xgb, label='Previsão XGBoost', alpha=0.7)
plt.legend()
plt.title('Previsão XGBoost vs Real')
plt.show()
"""))

nb.cells.append(nbf.v4.new_markdown_cell("""## 7. Conclusão
Comparação de desempenho dos modelos.
"""))

nb.cells.append(nbf.v4.new_code_cell("""results = pd.DataFrame({
    'Modelo': ['AutoARIMA', 'XGBoost'],
    'RMSE': [rmse_arima, rmse_xgb]
})
print(results)

best_model = results.loc[results['RMSE'].idxmin()]
print(f"\\nO melhor modelo é {best_model['Modelo']} com RMSE {best_model['RMSE']:.4f}")
"""))

with open('../notebooks/Forecasting_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print("Criado ../notebooks/Forecasting_Pipeline.ipynb")
