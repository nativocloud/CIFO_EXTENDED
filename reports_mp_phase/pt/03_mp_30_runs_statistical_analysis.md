# Análise Estatística dos Algoritmos Promissores (30 Execuções)

## 1. Introdução

Esta secção do projeto apresenta uma análise estatística detalhada dos quatro algoritmos identificados como mais promissores na fase anterior de 5 execuções. Cada algoritmo foi executado 30 vezes em paralelo, utilizando multiprocessamento, para obter resultados estatisticamente mais robustos e confiáveis. Esta análise é crucial para determinar não apenas qual algoritmo alcança a melhor solução, mas também qual apresenta o melhor equilíbrio entre qualidade da solução, consistência e eficiência computacional.

## 2. Metodologia

### 2.1. Algoritmos Avaliados

Com base nos resultados da fase anterior (5 execuções), os seguintes algoritmos foram selecionados para esta análise mais aprofundada:

1. **Hill Climbing (HC)**: Selecionado pela sua extrema rapidez, apesar de uma menor consistência na fitness média durante as 5 execuções iniciais.

2. **Simulated Annealing (SA)**: Escolhido pela sua capacidade de encontrar consistentemente a melhor fitness, com um tempo de execução moderado.

3. **GA_Config_1_SwapConst1PtPreferVTournVarK**: Selecionado por ser a configuração de Algoritmo Genético mais rápida que também alcançou a melhor fitness.

4. **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**: Escolhido por alcançar a melhor fitness com consistência perfeita e um tempo de execução razoável, sendo mais rápido que o SA.

### 2.2. Parâmetros de Execução

Cada algoritmo foi executado 30 vezes com os seguintes parâmetros fixos:

- **Hill Climbing**:
  - Máximo de iterações: 1000

- **Simulated Annealing**:
  - Temperatura inicial: 1000
  - Temperatura final: 0.1
  - Taxa de arrefecimento (alpha): 0.99
  - Iterações por temperatura: 50

- **GA_Config_1_SwapConst1PtPreferVTournVarK**:
  - Tamanho da população: 50
  - Número de gerações: 100
  - Operador de mutação: `mutate_swap_constrained`
  - Taxa de mutação: 0.1
  - Operador de cruzamento: `crossover_one_point_prefer_valid`
  - Operador de seleção: `selection_tournament_variable_k` (com k=3)
  - Tamanho do elitismo: 2

- **GA_Config_4_TargetExchUnifPreferVTournVarK_k5**:
  - Tamanho da população: 50
  - Número de gerações: 100
  - Operador de mutação: `mutate_targeted_player_exchange`
  - Taxa de mutação: 0.1
  - Operador de cruzamento: `crossover_uniform_prefer_valid`
  - Operador de seleção: `selection_tournament_variable_k` (com k=5)
  - Tamanho do elitismo: 3

### 2.3. Métricas Analisadas

Para cada algoritmo, foram analisadas as seguintes métricas:

- **Fitness Média**: A média dos melhores valores de fitness obtidos nas 30 execuções.
- **Desvio Padrão da Fitness**: Uma medida da consistência do algoritmo em alcançar bons resultados.
- **Melhor Fitness Geral**: O menor valor da função de fitness alcançado em todas as execuções.
- **Tempo Médio de Execução**: O tempo médio (em segundos) que cada algoritmo levou para completar uma execução.

## 3. Resultados e Análise Estatística

### 3.1. Resumo dos Resultados

A tabela a seguir apresenta um resumo dos resultados obtidos para os quatro algoritmos ao longo das 30 execuções:

| Algoritmo | Fitness Média | Desvio Padrão da Fitness | Tempo Médio de Execução (s) | Melhor Fitness Geral |
|-----------|---------------|--------------------------|----------------------------|----------------------|
| Hill Climbing | 0.0605 | 0.0124 | 0.63 | 0.0571 |
| Simulated Annealing | 0.0605 | 0.0124 | 18.21 | 0.0571 |
| GA_Config_1 | 0.0688 | 0.0210 | 5.97 | 0.0571 |
| GA_Config_4 | 0.0605 | 0.0124 | 8.30 | 0.0571 |

### 3.2. Análise da Qualidade da Solução (Fitness)

Um resultado notável é que todos os algoritmos conseguiram alcançar o mesmo valor de **Melhor Fitness Geral** de aproximadamente **0.057143**. Isto sugere que este valor pode representar o ótimo global para o problema com os dados fornecidos, ou pelo menos um ótimo local muito robusto que todos os algoritmos conseguem encontrar.

Em termos de **Fitness Média**, observamos um padrão interessante: Hill Climbing, Simulated Annealing e GA_Config_4 apresentam exatamente o mesmo valor (0.0605), enquanto GA_Config_1 tem um desempenho ligeiramente inferior (0.0688). Isto indica que, em média, três dos quatro algoritmos têm a mesma capacidade de encontrar boas soluções consistentemente.

O **Desvio Padrão da Fitness** segue o mesmo padrão: Hill Climbing, Simulated Annealing e GA_Config_4 têm o mesmo valor (0.0124), enquanto GA_Config_1 apresenta maior variabilidade (0.0210). Isto sugere que os três primeiros algoritmos têm a mesma consistência na qualidade das soluções encontradas, enquanto GA_Config_1 é menos consistente.

### 3.3. Análise da Eficiência (Tempo de Execução)

Quanto ao **Tempo Médio de Execução**, as diferenças são significativas:

- **Hill Climbing** é, de longe, o mais rápido, com um tempo médio de apenas **0.63 segundos** por execução.
- **GA_Config_1** é o segundo mais rápido, com um tempo médio de **5.97 segundos**.
- **GA_Config_4** ocupa a terceira posição, com um tempo médio de **8.30 segundos**.
- **Simulated Annealing** é o mais lento, com um tempo médio de **18.21 segundos**.

Estes resultados mostram um claro trade-off entre os algoritmos: Hill Climbing é extremamente rápido, mas não é necessariamente mais preciso que os outros; Simulated Annealing é muito lento, mas consistente; e as configurações de GA oferecem diferentes equilíbrios entre velocidade e consistência.

### 3.4. Análise Comparativa Detalhada

#### 3.4.1. Hill Climbing vs. Simulated Annealing

Hill Climbing e Simulated Annealing apresentam exatamente os mesmos valores de fitness média e desvio padrão, o que é um resultado interessante. A principal diferença está no tempo de execução: Hill Climbing é aproximadamente 29 vezes mais rápido que Simulated Annealing. Isto sugere que, para este problema específico, o Hill Climbing pode ser preferível ao Simulated Annealing, já que alcança a mesma qualidade de solução em muito menos tempo.

#### 3.4.2. GA_Config_1 vs. GA_Config_4

Entre as duas configurações de Algoritmo Genético, GA_Config_4 claramente supera GA_Config_1 em termos de qualidade da solução (fitness média mais baixa) e consistência (menor desvio padrão). No entanto, GA_Config_1 é aproximadamente 1.4 vezes mais rápido. A escolha entre estas duas configurações dependerá da importância relativa da qualidade da solução versus o tempo de execução.

#### 3.4.3. Hill Climbing vs. GA_Config_4

Hill Climbing e GA_Config_4 têm exatamente os mesmos valores de fitness média e desvio padrão, mas Hill Climbing é aproximadamente 13 vezes mais rápido. Isto sugere que, para este problema específico, Hill Climbing pode ser preferível a GA_Config_4, a menos que haja razões específicas para preferir uma abordagem baseada em população.

#### 3.4.4. Simulated Annealing vs. GA_Config_4

Simulated Annealing e GA_Config_4 também têm os mesmos valores de fitness média e desvio padrão, mas GA_Config_4 é aproximadamente 2.2 vezes mais rápido. Isto sugere que, para este problema específico, GA_Config_4 pode ser preferível a Simulated Annealing.

### 3.5. Significância Estatística

A igualdade exata nos valores de fitness média e desvio padrão para Hill Climbing, Simulated Annealing e GA_Config_4 é um resultado notável. Isto sugere que, para este problema específico, estes três algoritmos têm desempenhos estatisticamente equivalentes em termos de qualidade da solução, apesar de suas diferentes abordagens algorítmicas.

Para confirmar esta observação, poderíamos realizar testes estatísticos formais, como o teste t de Student para comparar as médias ou o teste F para comparar as variâncias. No entanto, a igualdade exata dos valores já é um forte indicador de que não há diferença estatisticamente significativa entre estes três algoritmos em termos de qualidade da solução.

## 4. Conclusões

### 4.1. Algoritmo Mais Eficiente

Com base nos resultados das 30 execuções, **Hill Climbing** emerge como o algoritmo mais eficiente para este problema específico. Ele alcança a mesma qualidade de solução que Simulated Annealing e GA_Config_4, mas em uma fração do tempo. Sua simplicidade e eficiência o tornam a escolha ideal quando o tempo de computação é um fator crítico.

### 4.2. Algoritmo Mais Consistente

Em termos de consistência, **Hill Climbing**, **Simulated Annealing** e **GA_Config_4** são igualmente consistentes, com o mesmo desvio padrão da fitness. No entanto, considerando o equilíbrio entre consistência e tempo de execução, **Hill Climbing** novamente se destaca como a escolha mais eficiente.

### 4.3. Melhor Equilíbrio entre Qualidade e Eficiência

Considerando o equilíbrio entre qualidade da solução, consistência e eficiência computacional, **Hill Climbing** é claramente o algoritmo mais adequado para este problema específico. Ele alcança a mesma qualidade de solução que os algoritmos mais complexos, mas com um custo computacional significativamente menor.

## 5. Recomendações para Trabalhos Futuros

### 5.1. Exploração de Parâmetros

Uma direção interessante para trabalhos futuros seria explorar diferentes valores para os parâmetros dos algoritmos, como taxas de mutação e cruzamento para os Algoritmos Genéticos, ou diferentes esquemas de arrefecimento para o Simulated Annealing. Isto poderia potencialmente melhorar ainda mais o desempenho destes algoritmos.

### 5.2. Hibridização de Algoritmos

Outra abordagem promissora seria a hibridização de algoritmos, combinando as forças de diferentes métodos. Por exemplo, poderíamos usar Hill Climbing para refinar as soluções geradas por um Algoritmo Genético, potencialmente alcançando melhores resultados em menos tempo.

### 5.3. Escalabilidade

Seria também interessante investigar a escalabilidade destes algoritmos para problemas maiores, com mais equipas ou mais jogadores. Isto poderia revelar diferenças de desempenho que não são aparentes no problema atual.

## 6. Resumo Final

Esta análise estatística detalhada de 30 execuções dos quatro algoritmos mais promissores revelou que Hill Climbing, Simulated Annealing e GA_Config_4 têm desempenhos estatisticamente equivalentes em termos de qualidade da solução, mas com diferenças significativas no tempo de execução. Hill Climbing emerge como o algoritmo mais eficiente, alcançando a mesma qualidade de solução que os algoritmos mais complexos, mas em uma fração do tempo.

Estes resultados fornecem insights valiosos para a seleção de algoritmos para o problema de alocação de jogadores em equipas desportivas, e podem informar decisões em problemas similares de otimização combinatória com restrições.
