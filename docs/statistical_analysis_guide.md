# Análise Estatística Comparativa: Testes Pareados vs. Não Pareados

Este documento explica a implementação e interpretação dos testes estatísticos pareados e não pareados para comparação de algoritmos de otimização, conforme implementado no notebook `01_single_processor_analysis.py`.

## Métricas Analisadas

Implementamos análise estatística para as seguintes métricas:

1. **Best Fitness Value**: O melhor valor de fitness encontrado em todas as execuções
2. **Average Fitness**: A média dos valores de fitness entre execuções
3. **Standard Deviation**: O desvio padrão dos valores de fitness
4. **Convergence Speed**: Velocidade de convergência (iterações até estabilização)
5. **Execution Time**: Tempo de execução de cada algoritmo
6. **Success Rate**: Taxa de sucesso em encontrar soluções válidas
7. **Fitness Over Time Curve**: Curva de evolução do fitness ao longo do tempo

## Testes Estatísticos Implementados

### Para Dois Algoritmos

- **Dados Pareados**: Teste de Wilcoxon (Wilcoxon signed-rank test)
- **Dados Não Pareados**: Teste U de Mann-Whitney (Mann-Whitney U test)

### Para Três ou Mais Algoritmos

- **Dados Pareados**: Teste de Friedman seguido por teste post-hoc de Nemenyi
- **Dados Não Pareados**: Teste de Kruskal-Wallis seguido por teste post-hoc de Dunn

## Dados Pareados vs. Não Pareados

### Dados Pareados

Os dados são considerados pareados quando:
- Todos os algoritmos são executados com as mesmas sementes aleatórias
- Cada execução correspondente de diferentes algoritmos usa a mesma semente
- Todos os algoritmos têm o mesmo número de execuções

Para garantir dados pareados, definimos `USE_FIXED_SEEDS = True` e usamos a lista `RANDOM_SEEDS` para controlar a aleatoriedade.

### Dados Não Pareados

Os dados são considerados não pareados quando:
- Os algoritmos são executados com sementes aleatórias diferentes
- Não há correspondência direta entre execuções de diferentes algoritmos

## Interpretação dos Resultados

### Hipóteses Estatísticas

- **Hipótese Nula (H₀)**: O desempenho dos algoritmos é igual para a métrica analisada
- **Hipótese Alternativa (H₁)**: Pelo menos um algoritmo tem desempenho diferente

### Significância Estatística

- Um valor p < 0.05 indica diferença estatisticamente significativa
- Para múltiplos algoritmos, testes post-hoc identificam quais pares específicos diferem

### Comparação Pareado vs. Não Pareado

O notebook gera uma tabela comparativa (`paired_vs_unpaired_comparison.csv`) que mostra:
- Se os testes pareados e não pareados concordam ou discordam
- Quais métricas mostram resultados diferentes entre os dois tipos de testes
- Taxa de concordância entre os métodos

## Visualizações

Para cada métrica, geramos:
- Boxplots comparativos com pontos individuais
- Tabelas de resumo com valores médios
- Gráficos de convergência normalizados

## Recomendações para Análise Robusta

1. Execute cada algoritmo múltiplas vezes (NUM_RUNS ≥ 30)
2. Use sementes aleatórias consistentes para testes pareados
3. Interprete os resultados no contexto do domínio do problema
4. Considere tanto significância estatística quanto magnitude das diferenças
5. Verifique se há concordância entre testes pareados e não pareados

## Outputs Gerados

- `metrics_summary.csv`: Resumo de todas as métricas por algoritmo
- `paired_vs_unpaired_comparison.csv`: Comparação entre resultados pareados e não pareados
- `*_comparison.png`: Visualizações comparativas para cada métrica
- `statistical_comparison_paired.json` e `statistical_comparison_unpaired.json`: Resultados detalhados dos testes
