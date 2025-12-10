import pandas as pd

try:
    df = pd.read_csv("Energia-IA/100k - Medidor 120/energia_202512091732-100k-120.csv", parse_dates=['data'], index_col='data')
    df = df[['potencia_total_3_fases']].rename(columns={'potencia_total_3_fases': 'potencia_watts'})
    df = df.resample('15min').mean().interpolate()
    
    # Days
    df['day'] = df.index.day_name()
    best_day = df.groupby('day')['potencia_watts'].mean().idxmax()
    
    # Hours
    df['hour'] = df.index.hour
    best_hour = df.groupby('hour')['potencia_watts'].mean().idxmax()
    
    print(f"RESULT_DAY:{best_day}")
    print(f"RESULT_HOUR:{best_hour}")

except Exception as e:
    print(f"ERR:{e}")
