# Análise Estatística dos Resultados da Variação de Parâmetros (30 Execuções por Variação)

Esta secção detalha a análise estatística dos resultados obtidos após a execução de 30 repetições para cada uma das 19 variações de parâmetros aplicadas ao Hill Climbing (HC) e às duas configurações de Algoritmos Genéticos (GA_Config_1 e GA_Config_4). O objetivo desta fase foi identificar as configurações de hiperparâmetros mais eficazes para cada algoritmo, focando na qualidade da solução (fitness média), consistência (desvio padrão da fitness) e eficiência computacional (tempo médio de execução).

Todos os resultados e gráficos de convergência foram guardados no diretório `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/final_param_var_results/`, e o sumário consolidado encontra-se em `all_algorithms_summary_final_param_var_30runs.csv`.

## Observação Chave Global

Um resultado notável desta extensa fase de variação de parâmetros é que **todas as 19 configurações testadas, ao longo das suas 30 execuções, foram capazes de encontrar pelo menos uma vez a mesma melhor fitness global de 0.0571**. Isto sugere que este valor pode representar um ótimo global ou, pelo menos, um ótimo local muito forte e consistentemente alcançável pelos algoritmos e suas variações testadas.

## Análise Detalhada por Algoritmo

### 1. Hill Climbing (HC)

Para o Hill Climbing, variou-se o número máximo de iterações. As configurações testadas e os seus desempenhos médios ao longo de 30 execuções foram:

*   **HC_Iter_500 (Máx. Iterações = 500):**
    *   Fitness Média: 0.0588
    *   Desvio Padrão da Fitness: 0.0089
    *   Tempo Médio de Execução: 0.41 segundos
*   **HC_Iter_1000 (Máx. Iterações = 1000 - Base):**
    *   Fitness Média: 0.0605
    *   Desvio Padrão da Fitness: 0.0124
    *   Tempo Médio de Execução: 0.42 segundos
*   **HC_Iter_1500 (Máx. Iterações = 1500):**
    *   Fitness Média: 0.0638
    *   Desvio Padrão da Fitness: 0.0169
    *   Tempo Médio de Execução: 0.41 segundos

**Conclusão para Hill Climbing:**
A configuração `HC_Iter_500` demonstrou ser a mais eficaz entre as variações testadas. Conseguiu a melhor fitness média (0.0588) e o menor desvio padrão (0.0089), indicando boa qualidade e consistência da solução, tudo isto mantendo o tempo de execução extremamente baixo (0.41s). Aumentar o número de iterações para 1000 ou 1500 não resultou numa melhoria da fitness média ou da melhor fitness global encontrada, e, na verdade, levou a uma ligeira deterioração da fitness média e a um aumento da variabilidade. Portanto, 500 iterações parecem ser suficientes para o HC explorar eficazmente o espaço de procura neste problema.

### 2. Algoritmo Genético - Configuração 1 (GA_Config_1)

Esta configuração utiliza o operador de mutação `mutate_swap_constrained`, cruzamento `crossover_one_point_prefer_valid` e seleção por torneio com k=3. Os parâmetros base eram: Tamanho da População = 50, Número de Gerações = 100, Taxa de Mutação = 0.1.

Resultados das variações (fitness média, desvio padrão, tempo médio):

*   **GA_Config_1_Base:** (Pop=50, Gen=100, MutRate=0.1)
    *   Fitness Média: 0.0638, Desvio Padrão: 0.0169, Tempo: 5.91s
*   **Variações da Taxa de Mutação (MutRate):**
    *   `MutRate=0.05`: Fitness Média: 0.0688, Desvio Padrão: 0.0210, Tempo: 5.78s (Piorou)
    *   `MutRate=0.15`: Fitness Média: 0.0605, Desvio Padrão: 0.0124, Tempo: 5.99s (Melhorou)
    *   `MutRate=0.25`: Fitness Média: 0.0588, Desvio Padrão: 0.0089, Tempo: 6.13s (**Melhor variação de MutRate para GA1**)
*   **Variações do Tamanho da População (PopSize):**
    *   `PopSize=30`: Fitness Média: 0.0732, Desvio Padrão: 0.0251, Tempo: 3.49s (Piorou, mas mais rápido)
    *   `PopSize=75`: Fitness Média: 0.0588, Desvio Padrão: 0.0089, Tempo: 9.08s (**Igualou melhor MutRate em qualidade/consistência, mas mais lento**)
*   **Variações do Número de Gerações (NumGen):**
    *   `NumGen=75`: Fitness Média: 0.0682, Desvio Padrão: 0.0228, Tempo: 4.47s (Piorou, mas mais rápido)
    *   `NumGen=150`: Fitness Média: 0.0605, Desvio Padrão: 0.0124, Tempo: 9.10s (Melhorou, mas mais lento)

**Conclusão para GA_Config_1:**
As variações mais promissoras para `GA_Config_1` foram `GA_Config_1_mutation_rate_0.25` (Pop=50, Gen=100) e `GA_Config_1_population_size_75` (MutRate=0.1, Gen=100). Ambas alcançaram uma fitness média de 0.0588 e um desvio padrão de 0.0089, igualando a qualidade e consistência da melhor configuração do Hill Climbing. No entanto, os seus tempos de execução foram consideravelmente mais altos (6.13s e 9.08s, respetivamente). Entre estas duas, a variação com taxa de mutação de 0.25 foi mais eficiente em termos de tempo.

### 3. Algoritmo Genético - Configuração 4 (GA_Config_4)

Esta configuração utiliza o operador de mutação `mutate_targeted_player_exchange`, cruzamento `crossover_uniform_prefer_valid` e seleção por torneio com k=5. Os parâmetros base eram: Tamanho da População = 50, Número de Gerações = 100, Taxa de Mutação = 0.1.

Resultados das variações (fitness média, desvio padrão, tempo médio):

*   **GA_Config_4_Base:** (Pop=50, Gen=100, MutRate=0.1)
    *   Fitness Média: 0.0621, Desvio Padrão: 0.0149, Tempo: 8.45s
*   **Variações da Taxa de Mutação (MutRate):**
    *   `MutRate=0.05`: Fitness Média: 0.0621, Desvio Padrão: 0.0149, Tempo: 9.50s (Semelhante à base, mais lento)
    *   `MutRate=0.15`: Fitness Média: 0.0571, Desvio Padrão: ~0.0000, Tempo: 8.55s (**Excelente: fitness perfeita, consistência perfeita**)
    *   `MutRate=0.25`: Fitness Média: 0.0571, Desvio Padrão: ~0.0000, Tempo: 8.90s (**Excelente: fitness perfeita, consistência perfeita**)
*   **Variações do Tamanho da População (PopSize):**
    *   `PopSize=30`: Fitness Média: 0.0638, Desvio Padrão: 0.0169, Tempo: 4.80s (Piorou, mas mais rápido)
    *   `PopSize=75`: Fitness Média: 0.0571, Desvio Padrão: ~0.0000, Tempo: 12.91s (**Excelente: fitness perfeita, consistência perfeita, mas mais lento**)
*   **Variações do Número de Gerações (NumGen):**
    *   `NumGen=75`: Fitness Média: 0.0638, Desvio Padrão: 0.0169, Tempo: 6.40s (Piorou, mas mais rápido)
    *   `NumGen=150`: Fitness Média: 0.0571, Desvio Padrão: ~0.0000, Tempo: 12.44s (**Excelente: fitness perfeita, consistência perfeita, mas mais lento**)

**Conclusão para GA_Config_4:**
A `GA_Config_4` demonstrou um potencial notável com o ajuste de parâmetros. Várias das suas variações (`MutRate=0.15`, `MutRate=0.25`, `PopSize=75`, `NumGen=150`) conseguiram alcançar consistentemente a melhor fitness média de 0.0571 (igual à melhor fitness global encontrada) com um desvio padrão praticamente nulo. Isto indica uma convergência muito robusta para a solução de alta qualidade. 
Entre estas configurações de topo, a `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15) foi a mais eficiente em termos de tempo, alcançando este desempenho em 8.55 segundos em média.

## Conclusões Gerais da Variação de Parâmetros

1.  **Melhor Fitness Global Atingível:** Todos os algoritmos e suas variações foram capazes de encontrar a melhor fitness de 0.0571, sugerindo que este é um valor de referência robusto para o problema.

2.  **Hill Climbing Otimizado:** A configuração `HC_Iter_500` (500 iterações) destacou-se pela sua extrema eficiência (0.41s), alcançando uma excelente fitness média (0.0588) e boa consistência (desvio padrão 0.0089).

3.  **GA_Config_1 Otimizado:** A melhor variação foi `GA_Config_1_mutation_rate_0.25`, que igualou a qualidade e consistência do `HC_Iter_500` (fitness média 0.0588, desvio padrão 0.0089), mas com um custo computacional significativamente maior (6.13s).

4.  **GA_Config_4 Otimizado:** Esta configuração, especialmente com `MutRate=0.15` (ou 0.25), `PopSize=75` ou `NumGen=150`, demonstrou a capacidade de atingir consistentemente a melhor fitness média de 0.0571 com desvio padrão nulo. A variante `GA_Config_4_mutation_rate_0.15` foi a mais rápida entre estas (8.55s) a atingir este nível de perfeição.

**Recomendação Final de Algoritmos e Configurações:**

*   **Para Eficiência Máxima com Excelente Qualidade:** `HC_Iter_500` é a escolha clara. Oferece uma solução de alta qualidade de forma muito rápida e consistente.
*   **Para Garantia da Melhor Fitness com Consistência Perfeita:** Se o objetivo é atingir a fitness de 0.0571 com a máxima consistência, a `GA_Config_4_mutation_rate_0.15` (Pop=50, Gen=100, MutRate=0.15) é a melhor opção, embora seja consideravelmente mais lenta que o HC.

A escolha entre estas dependerá dos requisitos específicos do problema em termos de tempo de execução versus a necessidade de garantir a melhor pontuação possível em todas as execuções.

## Justificação da Escolha dos Algoritmos Genéticos para Variação de Parâmetros

As configurações `GA_Config_1` e `GA_Config_4` foram selecionadas para a fase de variação de parâmetros detalhada com base no seu desempenho promissor na ronda anterior de 30 execuções (com parâmetros base). 

*   `GA_Config_1_SwapConst1PtPreferVTournVarK` foi a configuração de AG mais rápida nessa fase anterior que também demonstrou capacidade de alcançar a melhor fitness, embora com menor consistência que outras.
*   `GA_Config_4_TargetExchUnifPreferVTournVarK_k5` destacou-se por alcançar a melhor fitness com uma consistência muito boa (desvio padrão baixo) e um tempo de execução que, embora superior ao HC, era competitivo face a outras configurações de AG mais lentas e menos consistentes. A sua combinação de operadores (mutação direcionada e cruzamento uniforme com preferência por válidos) sugeria um bom potencial para exploração e explotação do espaço de soluções.

A expectativa era que, através da variação de parâmetros como taxa de mutação, tamanho da população e número de gerações, pudéssemos refinar ainda mais o desempenho destas duas arquiteturas de AG distintas, explorando diferentes equilíbrios entre exploração e explotação.

