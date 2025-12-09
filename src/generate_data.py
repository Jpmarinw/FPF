import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sensor_data(start_date, duration_days, freq='30s'):
    # Criar índice de data/hora
    dates = pd.date_range(start=start_date, periods=int(duration_days * 24 * 60 * 2) + 1, freq=freq)
    n = len(dates)
    
    # Consumo Base (Tendência) - levemente crescente
    trend = np.linspace(100, 120, n)
    
    # Sazonalidade Diária (24h) - Onda senoidal
    # 24 horas * 60 minutos * 2 (intervalos de 30s) = 2880 pontos por dia
    daily_seasonality = 20 * np.sin(2 * np.pi * np.arange(n) / (24 * 60 * 2))
    
    # Sazonalidade Semanal (7 dias) - Menor consumo nos fins de semana
    day_of_week = dates.dayofweek
    weekly_factor = np.where(day_of_week >= 5, 0.7, 1.0) # 30% de redução nos fins de semana
    
    # Ruído Aleatório
    noise = np.random.normal(0, 5, n)
    
    # Anomalias (Picos aleatórios)
    anomalies = np.zeros(n)
    n_anomalies = int(n * 0.001) # 0.1% de anomalias
    anomaly_indices = np.random.choice(n, n_anomalies, replace=False)
    anomalies[anomaly_indices] = np.random.choice([-30, 50], n_anomalies) # Quedas ou picos
    
    # Combinar componentes
    potencia = (trend + daily_seasonality) * weekly_factor + noise + anomalies
    potencia = np.maximum(potencia, 0) # Garantir que não haja consumo negativo
    
    df = pd.DataFrame({'timestamp': dates, 'potencia_watts': potencia})
    
    # Introduzir Gaps (Lacunas) - deletar 1% dos dados em blocos
    n_gaps = 10
    for _ in range(n_gaps):
        start_idx = np.random.randint(0, n - 100)
        gap_len = np.random.randint(10, 100)
        df.iloc[start_idx:start_idx+gap_len, 1] = np.nan
        
    return df

if __name__ == "__main__":
    start_date = datetime(2025, 1, 1)
    df = generate_sensor_data(start_date, duration_days=30) # 1 mês de dados
    df.to_csv("../data/sensor_data.csv", index=False)
    print(f"Gerado ../data/sensor_data.csv com {len(df)} linhas.")
