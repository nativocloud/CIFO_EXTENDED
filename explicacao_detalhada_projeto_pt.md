

## 6. Orquestração da Experiência e Análise de Resultados (`main_script.py` / `main.ipynb`)

O `main_script.py` (que é convertido para o notebook `main.ipynb`) é onde tudo se junta. Ele configura, executa e analisa os resultados dos diferentes algoritmos.

### 6.1. Configuração Inicial

*   **Importações**: Importa todas as classes e funções necessárias dos outros ficheiros (`solution.py`, `evolution.py`, `operators.py`), bem como bibliotecas como `pandas` (para manipulação de dados, especialmente tabelas), `numpy` (para cálculos numéricos como média e desvio padrão) e `matplotlib.pyplot` (para gráficos).
*   **Carregamento de Dados**: Lê o `players.csv` para um DataFrame do pandas e depois converte-o para uma lista de dicionários (`players`), que é o formato esperado pelos algoritmos.
*   **Parâmetros Globais**: Define constantes importantes:
    *   `NUM_TEAMS = 5`
    *   `TEAM_SIZE = 7`
    *   `MAX_BUDGET = 750`
    *   **`NUM_RUNS = 30`**: Este é um parâmetro CRUCIAL introduzido para obter resultados estatisticamente mais robustos para os algoritmos estocásticos (SA e AGs). Significa que cada uma destas abordagens será executada 30 vezes independentes.

### 6.2. Execução dos Algoritmos

O script executa cada algoritmo sequencialmente.

*   **Hill Climbing (HC)**:
    *   Executado **uma vez**, pois é determinístico para um ponto de partida (que é gerado aleatoriamente dentro da sua função, mas a busca em si é determinística).
    *   Guarda: a melhor solução (`hc_solution_obj`), o seu fitness (`hc_fitness_val`), o tempo de execução (`hc_exec_time`), e o histórico de convergência (`hc_history_convergence`).
    *   Gera um gráfico da convergência do HC.

*   **Simulated Annealing (SA)**:
    *   Executado **`NUM_RUNS` (30) vezes**.
    *   Um loop `for i in range(NUM_RUNS):` controla as execuções.
    *   Para cada execução:
        *   Chama `simulated_annealing_for_league()`.
        *   Guarda o fitness final e o tempo de execução dessa run específica em listas (`sa_all_fitness_values`, `sa_all_exec_times`).
        *   Mantém o registo da melhor solução global encontrada em todas as 30 runs (`best_sa_solution_overall`, `best_sa_fitness_overall`) e o seu histórico de convergência (`best_sa_history_overall`).
    *   Após o loop, calcula:
        *   `sa_mean_fitness`: Média dos melhores fitness das 30 runs.
        *   `sa_std_fitness`: Desvio padrão dos melhores fitness das 30 runs (mede a variabilidade/consistência do SA).
        *   `sa_mean_exec_time`: Tempo médio de execução por run.
    *   Gera um gráfico da convergência da *melhor run* do SA.

*   **Genetic Algorithms (GAs)**:
    *   São definidas várias configurações de AG (`ga_configs_new`), cada uma sendo um dicionário que especifica:
        *   `name`: Um nome descritivo para a configuração.
        *   `mutation_operator_func`, `crossover_operator_func`, `selection_operator_func`: As funções dos operadores a serem usados.
        *   Parâmetros específicos dos operadores (ex: `tournament_k`, `boltzmann_temp`).
    *   Para **cada configuração de AG** na lista `ga_configs_new`:
        *   É executada **`NUM_RUNS` (30) vezes** (um loop `for i in range(NUM_RUNS):` dentro do loop das configurações).
        *   Para cada uma dessas 30 execuções:
            *   Chama `genetic_algorithm()` com os operadores e parâmetros da configuração atual.
            *   Guarda o fitness final e o tempo de execução dessa run específica em listas temporárias para essa configuração (`config_all_fitness`, `config_all_exec_times`).
            *   Mantém o registo da melhor solução global encontrada nas 30 runs *para essa configuração específica* (`config_best_sol_overall`, `config_best_fitness_overall`) e o seu histórico de convergência (`config_best_history_overall`).
        *   Após as 30 runs para uma configuração, calcula:
            *   `mean_fit`, `std_fit`, `mean_time` para essa configuração.
        *   Estes resultados agregados (média, dp, melhor global da config) são guardados numa lista chamada `ga_results_summary`.
        *   O histórico de convergência da melhor run dessa configuração é guardado em `all_ga_convergence_histories`.
    *   Após todas as configurações de AG terem sido executadas 30 vezes cada:
        *   Gera um gráfico de convergência sobreposto, mostrando a curva da *melhor run* para cada configuração de AG.

*Comentário Pessoal sobre Execuções Múltiplas*: A decisão de executar os algoritmos estocásticos múltiplas vezes (30 runs) é fundamental para uma análise séria. Uma única execução de um AG ou SA pode dar um resultado muito bom ou muito mau por acaso. A média e o desvio padrão dão uma imagem muito mais fiel do desempenho típico e da consistência do algoritmo.

### 6.3. Análise Comparativa e Visualização

Esta é a secção onde os resultados de todos os algoritmos e configurações são reunidos e comparados.

*   **Tabela Comparativa**: 
    *   Os dados recolhidos (fitness, tempo, operadores) para HC, SA (médias e melhor global), e cada configuração de AG (médias e melhor global) são compilados numa lista de dicionários (`comparison_data`).
    *   Esta lista é convertida num DataFrame do `pandas` (`comparison_df`) para fácil visualização e manipulação.
    *   A tabela impressa mostra:
        *   `Algorithm`: Nome do algoritmo ou configuração do AG.
        *   `Mean Best Fitness`: Média do melhor fitness obtido nas 30 runs (para SA/AGs). Para HC, é o fitness da única run.
        *   `Std Dev Best Fitness`: Desvio padrão do melhor fitness (para SA/AGs). Para HC, é 0.
        *   `Mean Execution Time (s)`: Tempo médio de execução por run (para SA/AGs). Para HC, é o tempo da única run.
        *   `Overall Best Fitness`: O melhor fitness absoluto encontrado em todas as 30 runs (para SA/AGs).
        *   Nomes dos operadores usados (para AGs).

*   **Gráficos Comparativos**:
    1.  **Gráfico de Barras do Melhor Fitness Médio**: 
        *   Mostra uma barra para cada algoritmo/configuração, onde a altura da barra é o `Mean Best Fitness`.
        *   Crucialmente, inclui **barras de erro** (`yerr=comparison_df_sorted_fitness["Std Dev Best Fitness"]`). Estas barras de erro visualizam o desvio padrão, dando uma ideia da variabilidade do desempenho. Se as barras de erro de duas configurações se sobrepuserem muito, as suas diferenças de média podem não ser estatisticamente significativas.
        *   O eixo Y é "Lower is Better" (Menor é Melhor).
    2.  **Gráfico de Barras do Tempo Médio de Execução**:
        *   Mostra o `Mean Execution Time (s)` para cada algoritmo/configuração.

*Comentário Pessoal sobre Análise*: A combinação da tabela detalhada com os gráficos (especialmente o de fitness com barras de erro) é uma forma poderosa de comparar os algoritmos. Permite não só ver qual foi o "melhor" em média, mas também quão consistentes foram e quanto custaram em termos de tempo.

### 6.4. Discussão e Conclusão (no Notebook)

As secções de Markdown no final do notebook (`## 5. Discussion of Results` e `## 6. Conclusion`) são preenchidas com texto que interpreta os resultados apresentados na tabela e nos gráficos. Estas secções foram atualizadas para refletir a análise baseada nas múltiplas execuções, discutindo:

*   O desempenho do Hill Climbing (rápido, mas preso em ótimos locais).
*   O desempenho do Simulated Annealing (equilíbrio entre qualidade e tempo, variabilidade).
*   O desempenho das diferentes configurações do Algoritmo Genético (potencial para encontrar as melhores soluções, consistência, custo computacional, efeito dos operadores).
*   Comparações gerais, considerando as médias e desvios padrão.
*   Limitações (ex: número de runs, necessidade de testes estatísticos formais) e trabalho futuro.

## 7. Dicas para Rever o Código e Lógica

Ao rever este projeto, sugiro que preste atenção aos seguintes pontos:

1.  **Validade das Soluções**: Siga o fluxo de como uma solução é criada e validada. A função `LeagueSolution.is_valid()` é o polícia das restrições. Verifique se todas as restrições do problema estão corretamente implementadas lá.
2.  **Operadores e Restrições**: Analise como os operadores de mutação e cruzamento (especialmente as versões `_constrained` e `_prefer_valid`) tentam manter a validade. Pergunte-se: Eles são eficazes? Há cenários onde podem falhar em produzir soluções válidas frequentemente?
3.  **Fluxo dos Algoritmos**: Para cada algoritmo em `evolution.py`, tente seguir o fluxo lógico passo a passo. Como é que a solução evolui (HC/SA) ou como é que a população evolui (AG)?
4.  **Parâmetros**: Muitos parâmetros influenciam o desempenho (taxa de mutação, tamanho da população, parâmetros do SA, `k` do torneio, etc.). Os valores atuais foram definidos, mas a otimização destes parâmetros (tuning) é muitas vezes uma parte importante do desenvolvimento de meta-heurísticas.
5.  **Execuções Múltiplas e Estatísticas**: Entenda por que as execuções múltiplas são importantes para SA e AG. No `main_script.py`, veja como os resultados de cada run são armazenados e depois agregados (usando `numpy.mean` e `numpy.std`).
6.  **Interpretação dos Resultados**: Olhe para a tabela comparativa e os gráficos. Tente tirar as suas próprias conclusões antes de ler a secção de discussão no notebook. As barras de erro no gráfico de fitness são particularmente importantes para julgar se as diferenças entre algoritmos são robustas.
7.  **Potenciais Melhorias (Pensamento Crítico)**:
    *   Os operadores poderiam ser mais inteligentes ou mais eficientes na geração de soluções válidas?
    *   Existem outras estratégias de vizinhança para HC ou SA que poderiam ser exploradas?
    *   No AG, a forma como se lida com filhos inválidos (adicionar um aleatório válido) é a melhor? Poderiam existir estratégias de reparação?
    *   Como poderiam ser feitos testes estatísticos formais para comparar os algoritmos (ex: testes t, ANOVA)?

## 8. Conclusão da Explicação

Este projeto implementa uma abordagem sólida e comparativa para resolver um problema de otimização combinatória complexo. A utilização de múltiplos algoritmos, a adaptação de operadores genéticos para lidar com restrições, e a análise estatística baseada em múltiplas execuções são pontos fortes.

Espero que esta explicação detalhada seja útil para a sua revisão! Se tiver mais perguntas ou quiser aprofundar algum ponto específico, estou à disposição.

*Com os melhores cumprimentos,
Manus*

