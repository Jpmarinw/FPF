# Desafio de Previsão de Consumo de Energia Industrial

## Sobre a Atividade

Esta atividade consiste no desenvolvimento de um **pipeline de Machine Learning "End-to-End"** (de ponta a ponta) para prever o consumo de energia elétrica de uma máquina industrial. O objetivo é utilizar dados históricos de sensores para antecipar qual será a demanda energética futura.

## Para que serve na prática?

No mundo real, a previsão de séries temporais (Forecasting) aplicada à indústria tem valor imenso:

1.  **Eficiência Energética e Redução de Custos**:
    -   Muitas indústrias pagam tarifas mais caras em horários de pico. Prevendo o consumo, a fábrica pode planejar suas operações para horários mais baratos ou evitar ultrapassar contratos de demanda.
2.  **Planejamento Operacional**:
    -   Saber quanto a fábrica vai consumir ajuda a planejar a compra de insumos e a escala de turnos de trabalho.
3.  **Detecção de Anomalias e Manutenção Preditiva**:
    -   Se o modelo prevê um consumo X e a máquina consome 2X, isso pode indicar um defeito, falta de lubrificação ou desgaste de peças. Detectar isso cedo evita quebras catastróficas.
4.  **Sustentabilidade**:
    -   Otimizar o uso de energia reduz a pegada de carbono da operação.

## O que foi feito neste projeto?

O projeto simula o fluxo de trabalho de um Cientista de Dados real:

### 1. Coleta e Geração de Dados (`data/`)

Como não tínhamos os dados reais, geramos um dataset sintético (`sensor_data.csv`) que imita o comportamento real:

-   **Tendência**: O consumo sobe levemente (simulando desgaste ou aumento de produção).
-   **Sazonalidade**: Padrões que se repetem a cada 24h (dia/noite) e semanalmente (dias úteis vs fim de semana).
-   **Ruído e Anomalias**: Pequenas variações aleatórias e picos de erro.

### 2. Pré-processamento

Os dados brutos de sensores costumam ser "sujos".

-   **Reamostragem**: Transformamos dados de 30 segundos em médias de 15 minutos para reduzir o ruído e facilitar a análise.
-   **Limpeza**: Preenchemos falhas (gaps) na coleta de dados usando matemática (interpolação) para não deixar "buracos" no histórico.

### 3. Análise Exploratória (EDA) (`images/`)

Antes de tentar prever, precisamos entender. Criamos gráficos para responder:

-   "O consumo é constante?" (Não, varia com o tempo).
-   "Existe padrão diário?" (Sim, sobe de dia e cai à noite).
-   "A série é estacionária?" (Teste estatístico ADF).

### 4. Engenharia de Atributos (Feature Engineering)

Preparamos os dados para os modelos de IA.

-   **Transformação Cíclica**: Ensinamos ao modelo que 23h é perto de 00h usando Seno e Cosseno (já que para o computador, 23 e 0 são números distantes).
-   **Lags (Atrasos)**: Criamos colunas com o valor de "ontem" e "uma hora atrás", pois o passado recente é o melhor previsor do futuro imediato.

### 5. Modelagem (`notebooks/`)

Comparamos duas abordagens:

-   **AutoARIMA**: Um método estatístico clássico, muito bom para tendências simples, mas lento e às vezes limitado para dados complexos.
-   **XGBoost**: Um algoritmo de Machine Learning moderno baseado em Árvores de Decisão. Ele aprendeu padrões complexos e superou o ARIMA, conseguindo prever o consumo com muito mais precisão.

## Como usar este projeto

1.  Abra a pasta `notebooks/`.
2.  Execute o arquivo `Forecasting_Pipeline.ipynb` no Jupyter Notebook ou VS Code.
3.  Acompanhe o passo a passo da análise, desde os dados brutos até a conclusão de qual modelo é o melhor.
