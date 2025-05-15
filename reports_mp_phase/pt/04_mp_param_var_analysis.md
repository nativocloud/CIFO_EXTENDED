# Análise dos Resultados da Variação de Parâmetros para Algoritmos Genéticos

Este relatório detalha a análise dos resultados obtidos a partir da execução de experiências de variação de parâmetros para duas configurações promissoras de Algoritmos Genéticos (AG): GA_Config_1 e GA_Config_4. O objetivo foi identificar as configurações de hiperparâmetros que otimizam o desempenho destes algoritmos em termos de qualidade da solução (fitness média), consistência (desvio padrão da fitness) e eficiência (tempo médio de execução).

As variações testadas incidiram sobre a Taxa de Mutação (MutRate), Tamanho da População (PopSize) e Número de Gerações (NumGen), com cada variação a ser executada 5 vezes.

## Metodologia de Análise

Os dados foram recolhidos a partir do ficheiro `all_ga_variations_summary_mp_5runs.csv`. Para cada configuração base (GA_Config_1 e GA_Config_4), analisou-se o impacto de cada parâmetro variado, mantendo os outros nos seus valores base (PopSize=50, NumGen=100, MutRate=0.1, exceto quando o próprio parâmetro estava a ser variado).

## Resultados e Análise para GA_Config_1

A configuração base GA_Config_1 utiliza `mutate_swap_constrained`, `crossover_one_point_prefer_valid` e `selection_tournament_variable_k` (k=3).

### Impacto da Taxa de Mutação (MutRate) em GA_Config_1 (PopSize=50, NumGen=100)

| MutRate | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 0.05    | 0.0770       | 0.0244          | 5.68               |
| **0.10 (Base)** | **0.0688**   | **0.0210**      | **5.97**           | (Resultado das 30 execuções anteriores para referência)
| 0.15    | 0.0571       | 0.0000          | 6.07               |
| 0.25    | 0.0671       | 0.0199          | 6.31               |

**Observações para GA_Config_1 - MutRate:**
Uma taxa de mutação de 0.15 resultou na melhor fitness média (0.0571) e perfeita consistência (desvio padrão de 0.0), superando a taxa base de 0.1. Taxas mais baixas (0.05) ou mais altas (0.25) levaram a uma fitness média ligeiramente pior e menor consistência. O tempo de execução aumentou marginalmente com taxas de mutação mais altas.

### Impacto do Tamanho da População (PopSize) em GA_Config_1 (MutRate=0.1, NumGen=100)

| PopSize | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 30      | 0.0870       | 0.0244          | 3.48               |
| **50 (Base)** | **0.0688**   | **0.0210**      | **5.97**           |
| 75      | 0.0770       | 0.0244          | 9.02               |

**Observações para GA_Config_1 - PopSize:**
A população base de 50 indivíduos apresentou o melhor compromisso. Reduzir a população para 30 piorou significativamente a fitness média, embora tenha sido mais rápido. Aumentar para 75 não melhorou a fitness média em relação à base e aumentou consideravelmente o tempo de execução.

### Impacto do Número de Gerações (NumGen) em GA_Config_1 (MutRate=0.1, PopSize=50)

| NumGen | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|--------|--------------|-----------------|--------------------|
| 75     | 0.0571       | 0.0000          | 4.43               |
| **100 (Base)**| **0.0688**   | **0.0210**      | **5.97**           |
| 150    | 0.0770       | 0.0244          | 8.84               |

**Observações para GA_Config_1 - NumGen:**
Reduzir o número de gerações para 75 resultou na melhor fitness média (0.0571) e perfeita consistência, além de ser mais rápido que a configuração base. Aumentar o número de gerações para 150 não trouxe melhorias na fitness média e aumentou o tempo de execução.

### Melhor Variação para GA_Config_1:
Considerando as variações individuais, a `GA_Config_1_NumGen_75` (MutRate=0.1, PopSize=50, NumGen=75) e `GA_Config_1_MutRate_0.15` (MutRate=0.15, PopSize=50, NumGen=100) destacaram-se, ambas alcançando a fitness média de 0.0571 com desvio padrão de 0.0. A `GA_Config_1_NumGen_75` foi mais rápida (4.43s vs 6.07s).

## Resultados e Análise para GA_Config_4

A configuração base GA_Config_4 utiliza `mutate_targeted_player_exchange`, `crossover_uniform_prefer_valid` e `selection_tournament_variable_k` (k=5).

### Impacto da Taxa de Mutação (MutRate) em GA_Config_4 (PopSize=50, NumGen=100)

| MutRate | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 0.05    | 0.0671       | 0.0199          | 8.12               |
| **0.10 (Base)** | **0.0605**   | **0.0124**      | **8.30**           | (Resultado das 30 execuções anteriores para referência)
| 0.15    | 0.0571       | 0.0000          | 8.89               |
| 0.25    | 0.0571       | 0.0000          | 8.68               |

**Observações para GA_Config_4 - MutRate:**
Taxas de mutação de 0.15 e 0.25 resultaram na melhor fitness média (0.0571) e perfeita consistência, superando a taxa base de 0.1. O tempo de execução aumentou ligeiramente com estas taxas mais altas.

### Impacto do Tamanho da População (PopSize) em GA_Config_4 (MutRate=0.1, NumGen=100)

| PopSize | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|---------|--------------|-----------------|--------------------|
| 30      | 0.0571       | 0.0000          | 4.96               |
| **50 (Base)** | **0.0605**   | **0.0124**      | **8.30**           |
| 75      | 0.0571       | 0.0000          | 12.93              |

**Observações para GA_Config_4 - PopSize:**
Reduzir o tamanho da população para 30 melhorou a fitness média para 0.0571 com perfeita consistência e reduziu significativamente o tempo de execução (4.96s). Aumentar para 75 também alcançou a fitness ótima e consistência perfeita, mas com um custo de tempo de execução muito maior.

### Impacto do Número de Gerações (NumGen) em GA_Config_4 (MutRate=0.1, PopSize=50)

| NumGen | Mean Fitness | Std Dev Fitness | Mean Exec Time (s) |
|--------|--------------|-----------------|--------------------|
| 75     | 0.0571       | 0.0000          | 6.28               |
| **100 (Base)**| **0.0605**   | **0.0124**      | **8.30**           |
| 150    | 0.0571       | 0.0000          | 12.32              |

**Observações para GA_Config_4 - NumGen:**
Reduzir o número de gerações para 75 melhorou a fitness média para 0.0571 com perfeita consistência e reduziu o tempo de execução (6.28s). Aumentar para 150 também alcançou a fitness ótima e consistência, mas com um tempo de execução maior.

### Melhor Variação para GA_Config_4:
A variação `GA_Config_4_PopSize_30` (MutRate=0.1, PopSize=30, NumGen=100) foi a mais eficiente, alcançando a fitness ótima de 0.0571 com consistência perfeita no menor tempo de execução (4.96s). Outras variações como `GA_Config_4_NumGen_75` e aquelas com `MutRate` de 0.15 ou 0.25 também foram excelentes em termos de fitness e consistência, mas mais lentas.

## Conclusões da Variação de Parâmetros

1.  **Consistência Atingível:** Muitas variações, especialmente para GA_Config_4, conseguiram atingir a melhor fitness conhecida (0.0571) com perfeita consistência (desvio padrão 0.0) nas 5 execuções, o que é um resultado muito positivo.

2.  **GA_Config_1 Otimizada:** A configuração `GA_Config_1_NumGen_75` (MutRate=0.1, PopSize=50, NumGen=75) mostrou ser uma melhoria significativa em relação à base, alcançando fitness ótima e consistência perfeita com um tempo de execução de 4.43s.

3.  **GA_Config_4 Otimizada:** A configuração `GA_Config_4_PopSize_30` (MutRate=0.1, PopSize=30, NumGen=100) destacou-se como a mais eficiente para GA_Config_4, com fitness ótima, consistência perfeita e tempo de execução de 4.96s.

4.  **Impacto dos Parâmetros:**
    *   **Taxa de Mutação:** Para ambas as configurações, uma taxa de mutação ligeiramente superior à base (0.15) pareceu benéfica para atingir a melhor fitness consistentemente.
    *   **Tamanho da População:** Para GA_Config_4, reduzir a população para 30 foi vantajoso. Para GA_Config_1, a base de 50 ainda foi melhor que 30 ou 75.
    *   **Número de Gerações:** Reduzir o número de gerações para 75 foi benéfico para ambas as configurações, sugerindo que a convergência para a melhor solução pode ocorrer mais cedo do que as 100 gerações base, especialmente com outros parâmetros bem ajustados.

5.  **Comparação entre GA_Config_1 e GA_Config_4 Otimizadas:**
    *   `GA_Config_1_NumGen_75`: Fitness Média 0.0571, StdDev 0.0, Tempo 4.43s
    *   `GA_Config_4_PopSize_30`: Fitness Média 0.0571, StdDev 0.0, Tempo 4.96s

    Ambas as configurações otimizadas atingiram resultados idênticos em termos de qualidade e consistência da solução. A `GA_Config_1_NumGen_75` foi marginalmente mais rápida.

## Recomendações para Execuções Futuras

Com base nesta análise de variação de parâmetros, as seguintes configurações de AG são recomendadas para consideração futura, caso se opte por utilizar AGs em detrimento do Hill Climbing (que se mostrou mais eficiente na fase anterior):

*   **Opção 1 (Baseada em GA_Config_1):**
    *   Operadores: `mutate_swap_constrained`, `crossover_one_point_prefer_valid`, `selection_tournament_variable_k` (k=3)
    *   Parâmetros: PopSize=50, NumGen=75, MutRate=0.1 (ou 0.15 para possivelmente maior robustez, com ligeiro aumento de tempo)

*   **Opção 2 (Baseada em GA_Config_4):**
    *   Operadores: `mutate_targeted_player_exchange`, `crossover_uniform_prefer_valid`, `selection_tournament_variable_k` (k=5)
    *   Parâmetros: PopSize=30, NumGen=100 (ou 75), MutRate=0.1 (ou 0.15 / 0.25 para consistência, com aumento de tempo)

É importante notar que o Hill Climbing, na análise anterior de 30 execuções, já alcançava consistentemente a fitness de 0.0571 em cerca de 0.63 segundos. As configurações de AG otimizadas aqui, embora consistentes, ainda são consideravelmente mais lentas (4.43s - 4.96s). A escolha final dependerá do balanço desejado entre a garantia de exploração do AG e a eficiência do HC para este problema específico.

