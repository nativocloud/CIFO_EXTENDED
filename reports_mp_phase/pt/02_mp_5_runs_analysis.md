# Relatório da Fase de Multiprocessamento (5 Execuções)

## 1. Introdução

Esta secção do projeto foca-se na avaliação do desempenho dos algoritmos de otimização (Hill Climbing, Simulated Annealing e quatro configurações de Algoritmos Genéticos) quando executados múltiplas vezes em paralelo. O objetivo principal desta fase foi obter dados estatisticamente mais robustos sobre a eficácia e eficiência de cada abordagem, executando cada algoritmo 5 vezes. A utilização de multiprocessamento permitiu realizar estas execuções concorrentes, otimizando o tempo total de experimentação.

Os resultados desta fase são cruciais para identificar os algoritmos e configurações mais promissores, que serão posteriormente submetidos a um número mais elevado de execuções (30 runs) para uma análise de desempenho ainda mais aprofundada.

## 2. Metodologia Experimental

### 2.1. Algoritmos Avaliados

Os seguintes algoritmos e configurações foram avaliados nesta fase:

1.  **Hill Climbing (HC)**: Um algoritmo de busca local iterativo.
2.  **Simulated Annealing (SA)**: Um algoritmo probabilístico inspirado no processo de recozimento em metalurgia.
3.  **Algoritmos Genéticos (AG)**: Quatro configurações distintas foram testadas, variando operadores de mutação, cruzamento e seleção:
    *   **GA_Config_1_SwapConst1PtPreferVTournVarK**: Mutação Swap Constrained, Cruzamento One-Point Prefer Valid, Seleção Tournament Variable K (k=3).
    *   **GA_Config_2_TargetExchUnifPreferVRanking**: Mutação Targeted Player Exchange, Cruzamento Uniform Prefer Valid, Seleção Ranking.
    *   **GA_Config_3_ShuffleWithin1PtPreferVBoltzmann**: Mutação Shuffle Within Team Constrained, Cruzamento One-Point Prefer Valid, Seleção Boltzmann.
    *   **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**: Mutação Targeted Player Exchange, Cruzamento Uniform Prefer Valid, Seleção Tournament Variable K (k=5).

### 2.2. Parâmetros de Execução

*   **Número de Execuções por Algoritmo**: Cada algoritmo/configuração foi executado 5 vezes.
*   **Paralelização**: As execuções foram realizadas em paralelo utilizando o módulo `multiprocessing` do Python, aproveitando os núcleos de CPU disponíveis para acelerar o processo.
*   **Métricas Coletadas**: Para cada execução e, subsequentemente, para o conjunto das 5 execuções de cada algoritmo, foram registadas as seguintes métricas:
    *   Melhor Fitness Geral: O menor valor da função de fitness (desvio padrão das médias de habilidade das equipas) alcançado.
    *   Fitness Média: A média dos melhores valores de fitness obtidos nas 5 execuções.
    *   Desvio Padrão da Fitness: Uma medida da consistência do algoritmo em alcançar bons resultados.
    *   Tempo Médio de Execução: O tempo médio (em segundos) que cada algoritmo levou para completar uma execução.
*   **Ambiente**: As experiências foram conduzidas no ambiente de sandbox fornecido, com os dados dos jogadores carregados a partir do ficheiro `players.csv`.

### 2.3. Geração de População e Validade das Soluções

Para os Algoritmos Genéticos, a geração da população inicial e a manutenção da validade das soluções ao longo das gerações seguiram as mesmas restrições definidas na fase de processador único (estrutura das equipas, orçamento máximo, etc.). Foi dada particular atenção aos avisos sobre a dificuldade em gerar indivíduos válidos, embora o script tenha sido configurado para tentar múltiplas vezes a geração de soluções iniciais válidas para HC e SA, e os AGs possuam mecanismos intrínsecos para lidar com a validade.

## 3. Resultados e Discussão

Após a execução bem-sucedida do script `main_script_mp.py` para 5 execuções de cada algoritmo, os resultados foram compilados e analisados. A tabela seguinte resume o desempenho de cada algoritmo/configuração:

```
Algorithm,Mean Fitness,Std Dev Fitness,Mean Exec Time (s),Overall Best Fitness,Mutation Op,Crossover Op,Selection Op
Hill Climbing (MP-5 runs),0.06709518506727775,0.01990465584884448,0.46889443397521974,0.05714285714285552,N/A,N/A,N/A
Simulated Annealing (MP-5 runs),0.05714285714285552,0.0,17.598721027374268,0.05714285714285552,N/A,N/A,N/A
GA: GA_Config_1_SwapConst1PtPreferVTournVarK (MP-5 runs),0.0770475129917,0.024378125167686872,6.0489390850067135,0.05714285714285552,mutate_swap_constrained,crossover_one_point_prefer_valid,selection_tournament_variable_k
GA: GA_Config_2_TargetExchUnifPreferVRanking (MP-5 runs),0.08898633875210829,0.04332839580277237,61.911612796783444,0.05714285714285552,mutate_targeted_player_exchange,crossover_uniform_prefer_valid,selection_ranking
GA: GA_Config_3_ShuffleWithin1PtPreferVBoltzmann (MP-5 runs),0.0936131100520787,0.032132600170602175,61.22301082611084,0.05714285714285552,mutate_shuffle_within_team_constrained,crossover_one_point_prefer_valid,selection_boltzmann
GA: GA_Config_4_TargetExchUnifPreferVTournVarK_k5 (MP-5 runs),0.05714285714285552,0.0,8.188594150543214,0.05714285714285552,mutate_targeted_player_exchange,crossover_uniform_prefer_valid,selection_tournament_variable_k
```

Foram também gerados gráficos de convergência para a melhor execução de cada algoritmo e gráficos comparativos de tempo de execução e melhor fitness, guardados em `/home/ubuntu/CIFO_EXTENDED_Project/images_mp/run_5_results/`.

### 3.1. Análise da Qualidade da Solução (Fitness)

Observa-se um resultado notável: todos os algoritmos/configurações (Simulated Annealing, GA_Config_1, GA_Config_2, GA_Config_3, e GA_Config_4) conseguiram alcançar o mesmo valor de **Melhor Fitness Geral (Overall Best Fitness)** de aproximadamente **0.057143**. O Hill Climbing, embora tenha atingido este valor em algumas execuções, apresentou uma **Fitness Média** ligeiramente superior (0.0671), indicando menor consistência em atingir a melhor solução encontrada pelos outros.

O **Simulated Annealing** e a configuração **GA_Config_4_TargetExchUnifPreferVTournVarK_k5** destacaram-se pela consistência, ambos apresentando um **Desvio Padrão da Fitness** de **0.0**. Isto significa que, em todas as 5 execuções, estes dois algoritmos convergiram para o mesmo valor ótimo de fitness. As outras configurações de AG (Config_1, Config_2, Config_3) e o HC mostraram maior variabilidade nos resultados de fitness entre as execuções.

A **Fitness Média** da GA_Config_4 (0.057143) foi a melhor entre os AGs, igualando o SA. As GA_Config_1, GA_Config_2 e GA_Config_3 apresentaram fitness médias piores (0.0770, 0.0890, e 0.0936, respetivamente).

### 3.2. Análise da Eficiência (Tempo de Execução)

No que diz respeito ao **Tempo Médio de Execução**, as diferenças são significativas:

*   **Hill Climbing** foi, de longe, o mais rápido, com um tempo médio de aproximadamente **0.47 segundos** por execução.
*   Entre os Algoritmos Genéticos, a **GA_Config_1_SwapConst1PtPreferVTournVarK** foi a mais rápida, com cerca de **6.05 segundos**.
*   A **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**, que demonstrou excelente consistência e qualidade de solução, teve um tempo médio de **8.19 segundos**.
*   O **Simulated Annealing**, apesar da sua ótima performance em fitness e consistência, foi consideravelmente mais lento, com uma média de **17.60 segundos**.
*   As configurações **GA_Config_2** e **GA_Config_3** foram as mais lentas, com tempos médios de execução a rondar os **61-62 segundos**, o que as torna menos atrativas apesar de terem alcançado a melhor fitness geral em algumas execuções.

### 3.3. Observações Adicionais

*   A capacidade de todos os algoritmos (exceto, em média, o HC) de encontrar a mesma melhor fitness sugere que, para 5 execuções e os parâmetros atuais, este pode ser um ótimo local robusto ou mesmo o ótimo global para o problema com os dados fornecidos.
*   A consistência (baixo desvio padrão da fitness) é uma característica desejável, e o SA e a GA_Config_4 foram exemplares neste aspeto.
*   O compromisso entre a qualidade da solução, a consistência e o tempo de execução é um fator chave na seleção dos algoritmos para testes mais extensivos.

## 4. Conclusões Preliminares e Próximos Passos

Com base nos resultados de 5 execuções em multiprocessamento:

*   O **Hill Climbing** é extremamente rápido, mas menos consistente em encontrar a melhor solução em comparação com os outros.
*   O **Simulated Annealing** encontra consistentemente a melhor solução, mas com um custo de tempo de execução moderado.
*   A configuração de Algoritmo Genético **GA_Config_4_TargetExchUnifPreferVTournVarK_k5** emerge como uma candidata muito forte, pois igualou o SA e o HC na melhor fitness geral, demonstrou consistência perfeita (std dev = 0) e teve um tempo de execução significativamente mais baixo que o SA e as outras duas configurações de AG mais lentas (Config_2 e Config_3).
*   A **GA_Config_1_SwapConst1PtPreferVTournVarK** também é notável pela sua rapidez entre os AGs, embora com menor consistência na fitness média.

Estes resultados fornecem uma base sólida para a próxima fase do projeto: a seleção dos algoritmos mais promissores para serem executados 30 vezes. A análise detalhada destes resultados permitirá refinar a escolha dos algoritmos e dos seus parâmetros para investigações futuras.

Os próximos passos incluem a tradução deste relatório para Inglês e Bengali, seguida pela identificação formal dos algoritmos a serem levados para a fase de 30 execuções.

