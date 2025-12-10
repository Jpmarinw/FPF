import json

notebook_path = r'c:\Users\fpf\Documents\jpmarinw\semana2\FPF\notebooks\Consumption_Analysis.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define Forecast Section Cells
forecast_title_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Previsão de Demanda (Forecasting)\n",
        "Utilizando dados históricos para treinar um modelo de Machine Learning (XGBoost) capaz de prever o consumo futuro."
    ]
}

forecast_code_cell_1 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import xgboost as xgb\n",
        "from sklearn.metrics import mean_squared_error\n",
        "\n",
        "# Feature Engineering (Criar variáveis para o modelo)\n",
        "# Atributos Cíclicos (Hora e Dia da Semana)\n",
        "df_resampled['hour'] = df_resampled.index.hour\n",
        "df_resampled['day_of_week'] = df_resampled.index.dayofweek\n",
        "# Transformação Seno/Cosseno para manter a ciclicidade (23h próxima de 00h)\n",
        "df_resampled['hour_sin'] = np.sin(2 * np.pi * df_resampled['hour'] / 24)\n",
        "df_resampled['hour_cos'] = np.cos(2 * np.pi * df_resampled['hour'] / 24)\n",
        "\n",
        "# Lags (Valores passados)\n",
        "df_resampled['lag_1'] = df_resampled['potencia_watts'].shift(1) # Valor anterior (15min atrás)\n",
        "df_resampled['lag_24h'] = df_resampled['potencia_watts'].shift(96) # Valor de 24h atrás\n",
        "\n",
        "# Médias Móveis\n",
        "df_resampled['rolling_mean'] = df_resampled['potencia_watts'].rolling(window=4).mean()\n",
        "\n",
        "# Remover linhas com NaN gerados pelos Lags\n",
        "df_model = df_resampled.dropna()\n",
        "\n",
        "# Divisão Treino/Teste\n",
        "train_size = int(len(df_model) * 0.8)\n",
        "train, test = df_model.iloc[:train_size], df_model.iloc[train_size:]\n",
        "\n",
        "features = ['hour_sin', 'hour_cos', 'day_of_week', 'lag_1', 'lag_24h', 'rolling_mean']\n",
        "target = 'potencia_watts'\n",
        "\n",
        "X_train, y_train = train[features], train[target]\n",
        "X_test, y_test = test[features], test[target]\n",
        "\n",
        "print(f\"Treinando com {len(train)} amostras e testando com {len(test)} amostras...\")"
    ]
}

forecast_code_cell_2 = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Treinamento\n",
        "model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predição\n",
        "predictions = model.predict(X_test)\n",
        "rmse = np.sqrt(mean_squared_error(y_test, predictions))\n",
        "print(f\"Erro Médio (RMSE): {rmse:.2f} Watts\")\n",
        "\n",
        "# Visualização\n",
        "plt.figure(figsize=(15, 6))\n",
        "plt.plot(test.index, y_test, label='Real')\n",
        "plt.plot(test.index, predictions, label='Previsto', alpha=0.7)\n",
        "plt.title('Previsão de Consumo (XGBoost)')\n",
        "plt.ylabel('Potência (Watts)')\n",
        "plt.legend()\n",
        "plt.grid(True)\n",
        "plt.show()\n",
        "\n",
        "# Importância das Variáveis\n",
        "plt.figure(figsize=(10, 6))\n",
        "xgb.plot_importance(model, max_num_features=10)\n",
        "plt.title('Importância das Variáveis no Modelo')\n",
        "plt.show()"
    ]
}

nb['cells'].extend([forecast_title_cell, forecast_code_cell_1, forecast_code_cell_2])

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated with Forecast Analysis.")
