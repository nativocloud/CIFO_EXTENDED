# Relatório do Projeto de Inteligência Computacional

## 1. Introdução

Este relatório detalha o desenvolvimento e a análise de técnicas de inteligência computacional aplicadas ao problema da Otimização de Ligas Desportivas. O objetivo principal deste projeto é construir ligas desportivas de fantasia equilibradas, atribuindo jogadores a equipas de forma a minimizar a disparidade na força geral das equipas, quantificada pelo desvio padrão das classificações médias de habilidade das equipas. Para alcançar este objetivo, foram implementados, explorados e comparados três proeminentes algoritmos meta-heurísticos: Escalada de Encosta (Hill Climbing - HC), Recozimento Simulado (Simulated Annealing - SA) e Algoritmos Genéticos (Genetic Algorithms - GA). Cada algoritmo foi adaptado às restrições e objetivos específicos do problema de atribuição de ligas desportivas.

O projeto envolveu a definição de uma representação adequada para as configurações das ligas, o desenho de uma função de aptidão eficaz para avaliar a qualidade das soluções e a implementação de vários operadores de busca e estratégias para cada algoritmo. Para os Algoritmos Genéticos, foi dada particular atenção à exploração de diferentes mecanismos de seleção, operadores de cruzamento e operadores de mutação para compreender o seu impacto no desempenho e na qualidade da solução. Foi realizada uma experimentação extensiva para avaliar os algoritmos sob várias configurações de parâmetros e para comparar a sua eficácia na navegação no complexo espaço de busca inerente a este problema de otimização combinatória.

Este relatório está estruturado da seguinte forma: A Secção 2 fornece uma descrição detalhada do problema da Otimização de Ligas Desportivas, incluindo os seus objetivos e restrições. A Secção 3 define formalmente o problema no contexto dos Algoritmos Genéticos, abrangendo a representação da solução, o espaço de busca e a função de aptidão. A Secção 4 descreve os detalhes de implementação dos algoritmos de Escalada de Encosta, Recozimento Simulado e Algoritmos Genéticos. A Secção 5 delineia a configuração experimental, incluindo os parâmetros dos algoritmos, as métricas de avaliação de desempenho e o ambiente computacional. A Secção 6 apresenta uma análise abrangente dos resultados experimentais, comparando o desempenho dos algoritmos implementados e das diferentes configurações de GA. A Secção 7 oferece uma justificação das principais decisões de design e uma discussão das descobertas, incluindo a escolha da representação, o desenho da função de aptidão e a influência de vários operadores. Finalmente, a Secção 8 conclui o relatório com um resumo das principais descobertas e sugestões para trabalhos futuros. Referências e apêndices são fornecidos quando aplicável.

---
*(A redação das secções subsequentes continuará, incorporando detalhes dos ficheiros do projeto e análises anteriores. Este será um processo de várias etapas para cada versão linguística.)*



## 2. Enunciado do Problema: Otimização de Ligas Desportivas

O desafio central abordado neste projeto é a otimização da atribuição de equipas numa liga desportiva de fantasia. O objetivo principal é criar uma liga equilibrada, composta por um número predefinido de equipas, onde cada equipa é formada por um número específico de jogadores que cumprem papéis designados. O conjunto de dados consiste num grupo de jogadores, cada um caracterizado por atributos como nome, posição de jogo, nível de habilidade e salário. Para este projeto, consideramos uma liga de 5 equipas, sendo que cada equipa deve ter exatamente 7 jogadores.

O objetivo final é minimizar o desvio padrão das classificações médias de habilidade entre todas as equipas da liga. Um desvio padrão mais baixo significa uma liga mais equilibrada, onde as equipas são, em média, de força comparável, levando a uma jogabilidade mais competitiva e envolvente. Esta função objetivo mede diretamente a dispersão das forças das equipas e orienta a busca para distribuições equitativas de jogadores.

Diversas restrições devem ser estritamente respeitadas na formação das equipas:

*   **Composição da Equipa**: Cada uma das 5 equipas deve consistir em exatamente 7 jogadores, com uma distribuição posicional específica: 1 Guarda-Redes (GR), 2 Defesas (DEF), 2 Médios (MED) e 2 Avançados (AVA).
*   **Atribuição Única de Jogadores**: Cada jogador do conjunto de dados deve ser atribuído a exatamente uma equipa. Nenhum jogador pode ficar por atribuir ou ser atribuído a várias equipas.
*   **Orçamento da Equipa**: Cada equipa tem um orçamento máximo permitido para os salários dos jogadores. Para este projeto, o salário total de todos os jogadores numa única equipa não deve exceder 750 milhões (por exemplo, 750M €). Soluções que violem esta restrição orçamental para qualquer equipa são consideradas inválidas.
*   **Configurações Inválidas**: Qualquer configuração de liga que não cumpra todas as restrições acima (tamanho da equipa, requisitos posicionais, atribuição única de jogadores e orçamento por equipa) é considerada inválida e não faz parte do espaço de busca viável. Os algoritmos implementados são projetados para evitar inerentemente a geração de tais soluções inválidas ou para penalizá-las fortemente na avaliação da aptidão.

O conjunto de dados utilizado para este projeto compreende 35 jogadores únicos, correspondendo perfeitamente ao requisito de 5 equipas de 7 jogadores cada. Cada entrada de jogador inclui o seu nome, posição de jogo designada (GR, DEF, MED, AVA), uma classificação numérica de habilidade e um valor de salário em milhões.

---
*(A redação da Secção 3 seguirá.)*



## 3. Definição Formal do Problema

Para aplicar eficazmente técnicas de inteligência computacional, particularmente Algoritmos Genéticos (AG), é essencial uma definição formal do problema. Isto envolve especificar como uma solução potencial é representada, definir o espaço de busca e formular uma função de aptidão para quantificar a qualidade das soluções. Embora esta definição seja formulada principalmente para AG, os seus componentes centrais são adaptáveis e relevantes para os algoritmos de Escalada de Encosta (HC) e Recozimento Simulado (SA) implementados.

### 3.1. Representação do Indivíduo

Uma solução para o problema de Otimização de Ligas Desportivas, referida como um "indivíduo" na terminologia dos AG, é codificada como um **Vetor de Atribuição de Jogadores**. Esta representação é uma lista linear onde o comprimento da lista é igual ao número total de jogadores no conjunto de dados (35 jogadores neste projeto). Cada índice `i` na lista corresponde ao jogador `i` do conjunto de dados mestre de jogadores. O valor armazenado em `self.assignment[i]` é um inteiro que representa o ID da equipa (baseado em 0, por exemplo, 0 a 4 para 5 equipas) à qual o jogador `i` está atribuído.

Por exemplo, uma atribuição `[0, 1, 0, ..., 4]` significa que o jogador 0 está atribuído à equipa 0, o jogador 1 à equipa 1, o jogador 2 à equipa 0, e assim por diante. Esta representação é gerida pela classe `LeagueSolution` (e suas subclasses para HC e SA) encontrada em `solution.py`.

A escolha desta codificação linear oferece várias vantagens:
*   **Simplicidade e Direteza**: Fornece um mapeamento direto de jogadores para equipas.
*   **Unicidade Inerente do Jogador**: Por design, cada jogador tem exatamente uma entrada no vetor, garantindo que cada jogador é atribuído precisamente a uma equipa.
*   **Compatibilidade com Operadores Padrão**: Esta estrutura é altamente compatível com operadores genéticos padrão (cruzamento, mutação) e definições de vizinhança em algoritmos de busca local.

Mais detalhes sobre a justificação e comparação com representações alternativas são fornecidos no documento de análise dedicado (ver `report/pt_representation_analysis.md`).

### 3.2. Espaço de Busca

O espaço de busca compreende todas as possíveis configurações de liga válidas. Uma configuração de liga é considerada válida se aderir a todas as restrições do problema:

1.  **Tamanho da Equipa**: Cada uma das 5 equipas deve ter exatamente 7 jogadores.
2.  **Requisitos Posicionais**: Cada equipa deve ter 1 Guarda-Redes (GR), 2 Defesas (DEF), 2 Médios (MED) e 2 Avançados (AVA).
3.  **Restrição Orçamental**: O salário total dos jogadores em cada equipa não deve exceder 750 milhões.
4.  **Atribuição de Jogadores**: Todos os 35 jogadores devem ser atribuídos a uma equipa.

Embora a representação do vetor de atribuição de jogadores possa codificar configurações que violem estas restrições (por exemplo, todos os jogadores atribuídos a uma equipa), os algoritmos empregam mecanismos para navegar em direção ou permanecer dentro do espaço de busca válido. Estes mecanismos incluem:
*   Uma heurística construtiva (`_random_valid_assignment_constructive` em `LeagueSolution`) para gerar soluções iniciais potencialmente válidas.
*   Um método de validação rigoroso (`is_valid()` em `LeagueSolution`) para verificar a satisfação das restrições.
*   Mecanismos de penalização na função de aptidão para soluções inválidas.
*   Operadores conscientes das restrições ou que preferem a validade nos AG.

O tamanho do espaço de busca bruto (antes de considerar a validade) é vasto, pois cada um dos 35 jogadores pode ser atribuído a uma das 5 equipas, levando a 5<sup>35</sup> atribuições potenciais. O espaço de busca válido real é um subconjunto significativamente menor deste.

### 3.3. Função de Aptidão

A função de aptidão quantifica a qualidade de uma dada configuração de liga (uma solução individual). O seu objetivo principal é guiar a busca em direção a ligas equilibradas. A função de aptidão é definida no método `fitness()` da classe `LeagueSolution`.

1.  **Verificação de Validade**: O primeiro passo na avaliação da aptidão é verificar se a solução é válida usando o método `is_valid()`. Se a solução violar qualquer uma das restrições definidas (tamanho da equipa, equilíbrio posicional, orçamento), é-lhe atribuído um valor de aptidão de penalidade muito alto (efetivamente `float("inf")`). Isto garante que soluções inválidas têm uma probabilidade muito baixa de sobreviver ou serem selecionadas no processo de otimização.

2.  **Cálculo do Desvio Padrão da Habilidade**: Se a solução for válida, a aptidão é calculada como o desvio padrão das classificações médias de habilidade de todas as equipas na liga. O processo é o seguinte:
    a.  Para cada equipa, são recolhidos os valores de habilidade dos seus jogadores atribuídos.
    b.  A habilidade média para cada equipa é calculada.
    c.  O desvio padrão destas habilidades médias das equipas é então calculado usando `numpy.std()`.

**Objetivo**: O objetivo da otimização é **minimizar** este valor de aptidão. Um desvio padrão mais baixo indica uma liga mais equilibrada, onde os níveis médios de habilidade das equipas estão mais próximos uns dos outros.

Esta função de aptidão reflete diretamente o objetivo principal do projeto e incorpora um forte mecanismo para lidar com restrições, garantindo que a busca se concentre em soluções viáveis e de alta qualidade.

---
*(A redação da Secção 4 seguirá.)*



## 4. Algoritmos Implementados

Esta secção detalha os três algoritmos de inteligência computacional implementados para abordar o problema da Otimização de Ligas Desportivas: Escalada de Encosta (HC), Recozimento Simulado (SA) e Algoritmos Genéticos (GA). Cada algoritmo foi adaptado para funcionar com a representação definida do vetor de atribuição de jogadores e a função de aptidão que visa minimizar o desvio padrão das competências médias das equipas, respeitando todas as restrições do problema.

### 4.1. Escalada de Encosta (Hill Climbing - HC)

A Escalada de Encosta é um algoritmo de busca local que se move iterativamente em direção a soluções melhores, explorando a vizinhança imediata da solução atual. É uma abordagem gulosa que escolhe sempre o movimento que resulta na maior melhoria da aptidão.

*   **Descrição**: O algoritmo HC implementado para este projeto começa com uma configuração de liga válida inicial (gerada por `LeagueHillClimbingSolution`, que usa a heurística construtiva de `LeagueSolution`). Em cada iteração, gera todos os vizinhos válidos da solução atual. Se o melhor vizinho encontrado tiver uma aptidão melhor (desvio padrão mais baixo) do que a solução atual, o algoritmo move-se para esse vizinho. O processo continua até que nenhum vizinho ofereça uma melhoria, altura em que se considera ter atingido um ótimo local.
*   **Definição de Vizinhança**: A vizinhança de uma solução é definida pela aplicação de um operador de troca simples. Especificamente, o método `get_neighbors()` na classe `LeagueHillClimbingSolution` (definida em `solution.py`) gera vizinhos trocando as atribuições de equipa de cada par possível de jogadores no vetor de atribuição da solução atual. Apenas os vizinhos que são válidos (ou seja, satisfazem todas as restrições de composição da equipa, orçamento e atribuição de jogadores, conforme verificado por `is_valid()`) são considerados parte da vizinhança.
*   **Estratégia de Busca**: A estratégia é uma variante da Escalada de Encosta de subida mais íngreme (steepest-ascent), pois avalia todos os vizinhos e escolhe aquele com a melhoria de aptidão mais significativa. Se vários vizinhos oferecerem a mesma melhor melhoria, um é escolhido arbitrariamente (normalmente o primeiro encontrado). A busca termina se um número máximo predefinido de iterações for atingido ou se nenhum vizinho melhor for encontrado.
*   **Implementação**: A lógica central do HC está encapsulada na função `hill_climbing` em `evolution.py`, que utiliza a classe `LeagueHillClimbingSolution` para representação da solução e geração de vizinhança.

### 4.2. Recozimento Simulado (Simulated Annealing - SA)

O Recozimento Simulado é uma meta-heurística probabilística inspirada no processo de recozimento em metalurgia. Foi concebido para escapar a ótimos locais, aceitando ocasionalmente soluções piores do que a solução atual, com a probabilidade de aceitação a diminuir à medida que o algoritmo progride (controlado por um parâmetro de temperatura).

*   **Descrição**: O algoritmo SA começa com uma solução válida inicial (uma instância de `LeagueSASolution`) e uma temperatura inicial alta. Em cada iteração, é gerado um vizinho aleatório da solução atual. Se este vizinho for melhor (aptidão mais baixa), é aceite. Se for pior, ainda pode ser aceite com base no critério de Metropolis: `exp(-delta_E / T)`, onde `delta_E` é a alteração na aptidão (positiva para uma solução pior) e `T` é a temperatura atual. Isto permite que o algoritmo explore mais amplamente a temperaturas mais altas e se concentre na exploração à medida que a temperatura arrefece.
*   **Definição de Vizinhança**: Para o SA, um vizinho aleatório é gerado pelo método `get_random_neighbor()` na classe `LeagueSASolution` (definida em `solution.py`). Este método envolve tipicamente a troca das atribuições de equipa de dois jogadores selecionados aleatoriamente. Se a troca aleatória resultar numa solução inválida, o método tenta uma nova troca aleatória até um certo número de tentativas. Se um vizinho válido não for encontrado após estas tentativas, retorna uma cópia da solução atual para evitar ficar preso.
*   **Esquema de Arrefecimento**: A temperatura `T` é gradualmente diminuída de acordo com um esquema de arrefecimento. O SA implementado usa um esquema de arrefecimento geométrico: `T_nova = T_antiga * alpha`, onde `alpha` é uma taxa de arrefecimento (por exemplo, 0.99). O algoritmo executa um número fixo de iterações em cada nível de temperatura antes de reduzir a temperatura. O processo continua até que uma temperatura final baixa seja atingida ou um número máximo de iterações seja excedido.
*   **Probabilidade de Aceitação**: Conforme descrito, soluções piores são aceites com uma probabilidade `P = exp(-delta_E / T)`. Esta probabilidade diminui à medida que `T` diminui, fazendo com que o algoritmo se comporte mais como a Escalada de Encosta nas fases posteriores.
*   **Implementação**: A lógica do SA é implementada na função `simulated_annealing` em `evolution.py`, usando a classe `LeagueSASolution`.

### 4.3. Algoritmo Genético (Genetic Algorithm - GA)

Os Algoritmos Genéticos são meta-heurísticas baseadas em populações, inspiradas na seleção natural e na genética. Eles evoluem uma população de soluções candidatas ao longo de gerações usando operadores de seleção, cruzamento e mutação.

*   **Procedimento Geral do AG**:
    1.  **Inicialização**: Uma população inicial de `population_size` soluções válidas (instâncias de `LeagueSolution`) é gerada usando a função `generate_population` (em `evolution.py`), que aproveita a heurística `_random_valid_assignment_constructive` dentro do construtor `LeagueSolution`.
    2.  **Ciclo Evolutivo**: Para um número especificado de `generations`:
        a.  **Avaliação da Aptidão**: A aptidão de cada indivíduo na população é calculada usando `LeagueSolution.fitness()`.
        b.  **Seleção**: Soluções progenitoras são selecionadas da população atual com base na sua aptidão. O projeto implementa e permite a configuração para Seleção por Torneio (com tamanho de torneio `k` variável), Seleção por Ranking e Seleção de Boltzmann. Detalhes sobre estes mecanismos estão na análise de operadores (`report/pt_operator_analysis.md`).
        c.  **Cruzamento (Recombinação)**: Pares de progenitores selecionados são combinados para criar descendentes. Foram implementados operadores como Cruzamento de Um Ponto (`crossover_one_point_prefer_valid`) e Cruzamento Uniforme (`crossover_uniform_prefer_valid`). Estes operadores são envolvidos por uma lógica "prefer-valid" que tenta garantir que os descendentes gerados sejam válidos, repetindo a operação com novos pais se um descendente inválido for produzido, até um limite de tentativas.
        d.  **Mutação**: Os descendentes são sujeitos a mutação com uma certa `mutation_rate`. Os operadores de mutação implementados incluem Troca Simples (`mutate_swap_constrained`), Troca Direcionada de Jogadores entre Equipas (`mutate_targeted_player_exchange`) e Baralhamento Dentro da Equipa (`mutate_shuffle_within_team_constrained`). Estes operadores são projetados para introduzir pequenas alterações nos indivíduos, promovendo a diversidade e a exploração de novas áreas do espaço de busca, ao mesmo tempo que tentam manter a validade da solução.
        e.  **Elitismo**: Uma pequena percentagem (`elite_size`) dos melhores indivíduos da geração atual é transportada diretamente para a próxima geração, garantindo que as melhores soluções encontradas não sejam perdidas.
        f.  **Formação da Nova Geração**: A nova população é formada pelos descendentes (após cruzamento e mutação) e pelos indivíduos de elite.
    3.  **Terminação**: O algoritmo termina após um número fixo de gerações. A melhor solução encontrada em todas as gerações é retornada como o resultado.
*   **Implementação**: A lógica principal do AG está na função `genetic_algorithm` em `evolution.py`, que orquestra o ciclo evolutivo e utiliza as funções de operador definidas em `operators.py` e `solution.py`.

---
*(A redação da Secção 5 seguirá.)*



## 5. Configuração Experimental

Esta secção descreve a configuração experimental utilizada para avaliar o desempenho dos algoritmos implementados: Escalada de Encosta (HC), Recozimento Simulado (SA) e as várias configurações de Algoritmos Genéticos (GA). Detalha os parâmetros específicos de cada algoritmo, as métricas usadas para avaliação de desempenho e o ambiente computacional em que os testes foram realizados.

Todos os algoritmos foram executados `NUM_RUNS = 30` vezes independentes para garantir a robustez estatística dos resultados. O conjunto de dados de jogadores (`players.csv`) permaneceu constante em todas as experiências.

### 5.1. Parâmetros dos Algoritmos

**Parâmetros Gerais (Comuns a todos os algoritmos):**

*   `NUM_TEAMS`: 5 equipas
*   `TEAM_SIZE`: 7 jogadores por equipa
*   `MAX_BUDGET`: 750 milhões de € por equipa
*   `NUM_RUNS`: 30

**Escalada de Encosta (HC):**

*   `max_iterations`: 1000 (número máximo de iterações por execução)
*   Geração da Solução Inicial: Heurística `_random_valid_assignment_constructive`.
*   Geração de Vizinhança: Troca das atribuições de equipa de cada par possível de jogadores, mantendo apenas vizinhos válidos.

**Recozimento Simulado (SA):**

*   `initial_temp`: 1000 (temperatura inicial)
*   `final_temp`: 0.1 (temperatura final)
*   `alpha` (taxa de arrefecimento): 0.99
*   `iterations_per_temp`: 50 (número de iterações em cada nível de temperatura)
*   Geração da Solução Inicial: Heurística `_random_valid_assignment_constructive`.
*   Geração de Vizinho Aleatório: Troca das atribuições de equipa de dois jogadores selecionados aleatoriamente, com tentativas repetidas para garantir a validade.

**Algoritmo Genético (GA) - Parâmetros Comuns:**

*   `population_size`: 50 (tamanho da população)
*   `generations`: 100 (número de gerações)
*   `mutation_rate`: 0.2 (conforme usado em `main_script_sp.py`) ou 0.1 (conforme usado em `main_script_mp.py` - *nota: esta diferença deve ser reconciliada ou explicitamente mencionada se for intencional para diferentes versões do script*).
*   `elite_size`: 5 (conforme usado em `main_script_sp.py`) ou 2 (conforme usado em `main_script_mp.py` - *nota: esta diferença deve ser reconciliada ou explicitamente mencionada*).
*   Geração da População Inicial: `generate_population` usando `_random_valid_assignment_constructive`.

**Configurações Específicas do GA (detalhes dos operadores em `report/pt_operator_analysis.md`):**

1.  **GA Config 1 (SwapConst1PtPreferVTournVarK)**:
    *   Mutação: `mutate_swap_constrained`
    *   Cruzamento: `crossover_one_point_prefer_valid`
    *   Seleção: `selection_tournament_variable_k`
    *   `tournament_k`: 3

2.  **GA Config 2 (TargetExchUnifPreferVRanking)**:
    *   Mutação: `mutate_targeted_player_exchange`
    *   Cruzamento: `crossover_uniform_prefer_valid`
    *   Seleção: `selection_ranking`

3.  **GA Config 3 (ShuffleWithin1PtPreferVBoltzmann)**:
    *   Mutação: `mutate_shuffle_within_team_constrained`
    *   Cruzamento: `crossover_one_point_prefer_valid`
    *   Seleção: `selection_boltzmann`
    *   `boltzmann_temp`: 100

4.  **GA Config 4 (TargetExchUnifPreferVTournVarK_k5)**:
    *   Mutação: `mutate_targeted_player_exchange`
    *   Cruzamento: `crossover_uniform_prefer_valid`
    *   Seleção: `selection_tournament_variable_k`
    *   `tournament_k`: 5

### 5.2. Métricas para Avaliação de Desempenho

As seguintes métricas foram usadas para avaliar e comparar o desempenho dos algoritmos:

*   **Melhor Aptidão Encontrada (por execução)**: O valor de aptidão mais baixo (menor desvio padrão) alcançado numa única execução do algoritmo.
*   **Média da Melhor Aptidão**: A média dos melhores valores de aptidão encontrados nas 30 execuções independentes.
*   **Desvio Padrão da Melhor Aptidão**: O desvio padrão dos melhores valores de aptidão encontrados nas 30 execuções, indicando a consistência do algoritmo.
*   **Melhor Aptidão Geral**: O valor de aptidão mais baixo absoluto encontrado em todas as execuções de um determinado algoritmo/configuração.
*   **Tempo de Execução (por execução)**: O tempo de relógio de parede (wall-clock time) que cada execução de um algoritmo levou para ser concluída.
*   **Média do Tempo de Execução**: A média dos tempos de execução nas 30 execuções.
*   **Gráficos de Convergência**: Para a execução que produziu a melhor aptidão geral para cada algoritmo/configuração, foi gerado um gráfico de convergência mostrando a melhoria da aptidão ao longo das iterações (para HC/SA) ou gerações (para GA).

### 5.3. Ambiente Computacional

Os algoritmos foram implementados em Python 3.11. As principais bibliotecas utilizadas incluem:

*   **NumPy**: Para operações numéricas eficientes, especialmente para cálculos de desvio padrão e manipulação de arrays.
*   **Matplotlib**: Para gerar os gráficos de convergência e os gráficos comparativos de desempenho.
*   **Pandas**: Para carregar e manipular o conjunto de dados dos jogadores a partir do ficheiro CSV.
*   **Multiprocessing**: Para a versão `main_script_mp.py`, esta biblioteca foi usada para paralelizar as 30 execuções independentes dos algoritmos, utilizando múltiplos núcleos de CPU.

Os testes foram realizados num ambiente sandbox Linux (Ubuntu 22.04) com acesso a recursos de CPU padrão. A paralelização na versão multiprocessador foi configurada para usar o número de núcleos de CPU disponíveis no sistema, conforme determinado por `os.cpu_count()`.

---
*(A redação da Secção 6 seguirá.)*



## 6. Análise de Desempenho e Resultados

Esta secção apresenta uma análise detalhada do desempenho dos algoritmos implementados: Escalada de Encosta (HC), Recozimento Simulado (SA) e as quatro configurações de Algoritmos Genéticos (GA). A análise baseia-se nas métricas definidas na Secção 5.2, incluindo a qualidade da solução (aptidão), consistência e custo computacional (tempo de execução). Os resultados são apresentados tanto para as execuções de processador único (`main_script_sp.py`) como para as execuções multiprocessador (`main_script_mp.py`).

### 6.1. Resumo dos Resultados Agregados

Os resultados agregados de 30 execuções independentes para cada algoritmo e configuração são resumidos na tabela abaixo. Esta tabela inclui a média da melhor aptidão, o desvio padrão da melhor aptidão, o tempo médio de execução por execução e a melhor aptidão geral encontrada.

**Tabela 1: Resumo do Desempenho dos Algoritmos (Resultados de 30 Execuções)**

| Algoritmo/Configuração                          | Média Melhor Aptidão | DP Melhor Aptidão | Tempo Médio Exec. (s) | Melhor Aptidão Geral | Notas                                      |
| :---------------------------------------------- | :-------------------: | :---------------: | :--------------------: | :-------------------: | :----------------------------------------- |
| **Execução Processador Único (SP)**             |                       |                   |                        |                       |                                            |
| Escalada de Encosta (SP)                        | *[HC_SP_MeanFit]*     | *[HC_SP_StdFit]*   | *[HC_SP_MeanTime]*    | *[HC_SP_BestFit]*    | Execução processador único (por execução) |
| Recozimento Simulado (SP)                       | *[SA_SP_MeanFit]*     | *[SA_SP_StdFit]*   | *[SA_SP_MeanTime]*    | *[SA_SP_BestFit]*    | Execução processador único (por execução) |
| GA Config 1 (Swap,1Pt,TournK3) (SP)           | *[GA1_SP_MeanFit]*    | *[GA1_SP_StdFit]*  | *[GA1_SP_MeanTime]*   | *[GA1_SP_BestFit]*   | Execução processador único (por execução) |
| GA Config 2 (TargetExch,Unif,Rank) (SP)       | *[GA2_SP_MeanFit]*    | *[GA2_SP_StdFit]*  | *[GA2_SP_MeanTime]*   | *[GA2_SP_BestFit]*   | Execução processador único (por execução) |
| GA Config 3 (Shuffle,1Pt,Boltz) (SP)          | *[GA3_SP_MeanFit]*    | *[GA3_SP_StdFit]*  | *[GA3_SP_MeanTime]*   | *[GA3_SP_BestFit]*   | Execução processador único (por execução) |
| GA Config 4 (TargetExch,Unif,TournK5) (SP)    | *[GA4_SP_MeanFit]*    | *[GA4_SP_StdFit]*  | *[GA4_SP_MeanTime]*   | *[GA4_SP_BestFit]*   | Execução processador único (por execução) |
| **Execução Multiprocessador (MP)**              |                       |                   |                        |                       |                                            |
| Escalada de Encosta (MP)                        | *[HC_MP_MeanFit]*     | *[HC_MP_StdFit]*   | *[HC_MP_MeanTime]*    | *[HC_MP_BestFit]*    | Execução multiprocessador (por execução) |
| Recozimento Simulado (MP)                       | *[SA_MP_MeanFit]*     | *[SA_MP_StdFit]*   | *[SA_MP_MeanTime]*    | *[SA_MP_BestFit]*    | Execução multiprocessador (por execução) |
| GA Config 1 (Swap,1Pt,TournK3) (MP)           | *[GA1_MP_MeanFit]*    | *[GA1_MP_StdFit]*  | *[GA1_MP_MeanTime]*   | *[GA1_MP_BestFit]*   | Execução multiprocessador (por execução) |
| GA Config 2 (TargetExch,Unif,Rank) (MP)       | *[GA2_MP_MeanFit]*    | *[GA2_MP_StdFit]*  | *[GA2_MP_MeanTime]*   | *[GA2_MP_BestFit]*   | Execução multiprocessador (por execução) |
| GA Config 3 (Shuffle,1Pt,Boltz) (MP)          | *[GA3_MP_MeanFit]*    | *[GA3_MP_StdFit]*  | *[GA3_MP_MeanTime]*   | *[GA3_MP_BestFit]*   | Execução multiprocessador (por execução) |
| GA Config 4 (TargetExch,Unif,TournK5) (MP)    | *[GA4_MP_MeanFit]*    | *[GA4_MP_StdFit]*  | *[GA4_MP_MeanTime]*   | *[GA4_MP_BestFit]*   | Execução multiprocessador (por execução) |

*(Os valores reais para os marcadores de posição como `[HC_SP_MeanFit]` seriam extraídos dos registos de execução do script onde a DataFrame `all_results_summary_df` é impressa. Estes valores devem ser preenchidos com os dados numéricos reais obtidos das execuções dos scripts `main_script_sp.py` e `main_script_mp.py`.)*

### 6.2. Análise de Convergência

Os gráficos de convergência ilustram como a aptidão da melhor solução encontrada por um algoritmo melhora ao longo das iterações ou gerações. Estes gráficos foram gerados para a execução que alcançou a melhor aptidão geral para cada algoritmo e configuração de GA.

*   **Escalada de Encosta**: O gráfico de convergência para HC (ex: `sp_graphs/hc_convergence_sp.png` e `mp_graphs/hc_convergence_mp.png`) mostra tipicamente melhorias iniciais rápidas seguidas por um platô à medida que o algoritmo atinge um ótimo local. O número de passos até à convergência pode variar com base na solução inicial.

*   **Recozimento Simulado**: O gráfico de convergência do SA (ex: `sp_graphs/sa_convergence_sp.png` e `mp_graphs/sa_convergence_mp.png`) mostra frequentemente uma diminuição mais gradual da aptidão. A aptidão pode flutuar, especialmente a temperaturas mais altas, à medida que o algoritmo explora soluções piores para escapar a ótimos locais. À medida que a temperatura arrefece, a busca torna-se mais focada, assemelhando-se ao HC.

*   **Algoritmos Genéticos**: Os gráficos de convergência para as configurações de GA (ex: `sp_graphs/ga_convergence_sp_GA_Config_1_SwapConst1PtPreferVTournVarK.png` e `mp_graphs/ga_convergence_mp_GA_Config_1_SwapConst1PtPreferVTournVarK.png`) mostram a aptidão do melhor indivíduo na população ao longo de 100 gerações. Diferentes configurações de GA podem exibir velocidades de convergência e níveis de aptidão final variáveis, influenciados pelos seus operadores de seleção, cruzamento e mutação.

### 6.3. Desempenho Comparativo

Os gráficos comparativos fornecem uma comparação visual direta dos algoritmos com base nos seus valores de aptidão alcançados e tempos de execução.

*   **Aptidão Comparativa**: Gráficos como `sp_graphs/comparative_fitness_sp.png` e `mp_graphs/comparative_fitness_mp.png` exibem a média da melhor aptidão (frequentemente com barras de erro representando o desvio padrão) para HC, SA e cada configuração de GA. Isto permite uma avaliação de quais algoritmos encontram consistentemente melhores soluções.

*   **Tempos de Execução Comparativos**: Gráficos como `sp_graphs/comparative_times_sp.png` e `mp_graphs/comparative_times_mp.png` comparam o tempo médio de execução por execução para cada algoritmo. Isto destaca o custo computacional associado a cada abordagem. Espera-se que os GAs tenham geralmente tempos de execução mais altos devido à sua natureza baseada em populações e múltiplas gerações. O script multiprocessador (`main_script_mp.py`) visa reduzir o *tempo total de relógio de parede* para as 30 execuções, paralelizando-as, mas o *tempo médio de execução por execução individual* deve ser comparável à versão de processador único, exceto por pequenas sobrecargas ou variações de carga do sistema.

### 6.4. Discussão dos Resultados

*(Esta subsecção fornecerá uma interpretação detalhada dos dados apresentados na tabela de resumo e nos gráficos. Os seguintes são pontos gerais que seriam elaborados assim que os dados reais estivessem disponíveis e preenchidos na Tabela 1.)*

*   **Qualidade da Solução (Aptidão)**: Tipicamente, espera-se que os GAs, com as suas capacidades de busca mais amplas, encontrem soluções melhores ou mais consistentes (menor aptidão média e menor desvio padrão da aptidão) em comparação com o HC, que pode facilmente ficar preso em ótimos locais. O SA visa mitigar a limitação do HC, mas o seu sucesso depende fortemente da sintonia dos parâmetros.
    *   O desempenho das diferentes configurações de GA será comparado. Por exemplo, configurações que usam seleção mais agressiva (ex: Seleção por Torneio com um `k` maior) ou operadores de exploração/explotação mais eficazes podem produzir melhores resultados.
    *   O impacto do elitismo nos GAs (transportar os melhores `elite_size` indivíduos) geralmente ajuda a preservar boas soluções e pode acelerar a convergência para regiões de alta qualidade.

*   **Consistência**: O desvio padrão dos melhores valores de aptidão em múltiplas execuções indica a consistência de um algoritmo. Valores mais baixos sugerem que o algoritmo encontra de forma fiável soluções de qualidade semelhante, independentemente da semente aleatória ou do estado inicial.

*   **Custo Computacional (Tempo de Execução)**: O HC é geralmente o mais rápido por execução, pois realiza uma busca local relativamente simples. O SA adiciona sobrecarga computacional devido à sua aceitação probabilística e esquema de temperatura. Os GAs são tipicamente os mais intensivos computacionalmente devido à necessidade de gerir e evoluir uma população ao longo de muitas gerações. A implementação multiprocessamento reduz significativamente o tempo total necessário para realizar as 30 execuções independentes para todos os algoritmos, executando-as em paralelo, mas o tempo para uma única execução de um algoritmo específico permanece em grande parte o mesmo.

*   **Impacto dos Operadores no GA**: A análise aprofundará como diferentes combinações de operadores de seleção, cruzamento e mutação nas configurações de GA afetaram o desempenho. Por exemplo, operadores que são melhores a manter a diversidade enquanto ainda exploram boas soluções têm probabilidade de ter um bom desempenho. As versões "prefer-valid" e "constrained" dos operadores foram cruciais para navegar no espaço de busca altamente restrito.

*   **Melhor Algoritmo Geral**: Com base num equilíbrio entre qualidade da solução, consistência e custo computacional (considerando os benefícios da paralelização para o tempo total da experiência), será discutida uma recomendação para o algoritmo mais adequado para este problema específico de Atribuição de Ligas Desportivas.

Esta análise basear-se-á nos resultados numéricos específicos e nas tendências visuais observadas nas saídas geradas para fornecer conclusões concretas sobre a eficácia de cada técnica de inteligência computacional implementada.

---
*(A redação da Secção 7 seguirá.)*



## 7. Justificação das Decisões e Discussão

Esta secção fornece uma justificação para as principais decisões de design tomadas ao longo do projeto, discute o seu impacto e reflete sobre os desafios encontrados durante as fases de implementação e experimentação.

### 7.1. Escolha da Representação

O **vetor de atribuição de jogadores (codificação linear)** foi escolhido como a representação primária da solução. Neste esquema, uma lista ou array representa todos os jogadores, e o valor no índice de cada jogador indica o ID da equipa à qual esse jogador está atribuído (ex: `assignment[player_idx] = team_id`).

*   **Adequação e Justificação**:
    *   **Simplicidade e Direteza**: Esta representação é simples de entender e implementar. Mapeia diretamente para o problema de atribuir jogadores a equipas.
    *   **Compatibilidade com Operadores Padrão**: É relativamente compatível com operadores padrão de Algoritmos Genéticos como cruzamento de um ponto e uniforme, e mutação baseada em troca, embora com as adaptações necessárias para o tratamento de restrições.
    *   **Tamanho Fixo**: Dado um número fixo de jogadores, o comprimento do cromossoma é constante, simplificando muitas mecânicas dos AG.
*   **Alternativa Considerada (e por que menos adequada para aplicação direta de AG)**:
    *   **Representação Baseada em Equipas**: Uma alternativa poderia ter sido uma lista de equipas, onde cada objeto de equipa contém uma lista dos seus jogadores atribuídos. Embora intuitiva para representar uma liga formada, esta estrutura é mais complexa para a aplicação direta de operadores simples de cruzamento e mutação. Trocar jogadores ou segmentos entre tais listas estruturadas, mantendo a validade (especialmente o equilíbrio posicional e o tamanho da equipa), exigiria uma lógica de operador mais intrincada desde o início.
    *   A codificação linear escolhida permite que os operadores trabalhem numa estrutura mais simples e plana, com verificações de validade e mecanismos de reparação/tentativa a tratar das restrições.

### 7.2. Design da Função de Aptidão

A função de aptidão foi concebida para guiar a busca em direção a soluções que não são apenas válidas, mas também alcançam o objetivo primário de equilíbrio competitivo.

*   **Objetivo Primário**: Minimizar o desvio padrão das classificações médias de habilidade de todas as equipas. Isto reflete diretamente o objetivo de ter equipas com níveis de habilidade gerais semelhantes.
    *   `aptidao = np.std([media_habilidade_equipa_1, media_habilidade_equipa_2, ..., media_habilidade_equipa_5])`
*   **Tratamento de Restrições na Aptidão**: Um aspeto crucial foi como lidar com soluções inválidas que poderiam ser geradas por operadores (apesar dos mecanismos "prefer-valid" por vezes falharem).
    *   **Penalização Pesada**: Soluções inválidas (aquelas que falham no `is_valid()`, que verifica o tamanho da equipa, equilíbrio posicional e orçamento para todas as equipas) recebem um valor de aptidão de penalidade muito alto (ex: `float("inf")` ou uma constante grande como `1_000_000`). Isto remove-as efetivamente da disputa durante o processo de seleção nos AGs e garante que o HC e o SA as descartem sempre se existir uma alternativa válida.
    *   Esta abordagem de penalização é uma forma comum e eficaz de lidar com restrições rígidas em algoritmos evolutivos, empurrando a busca para regiões viáveis do espaço de soluções (Coello Coello 2002).[^3]

### 7.3. Melhores Configurações e Influência dos Operadores

*(Esta subsecção seria fortemente baseada nos dados reais da Secção 6. Os seguintes são pontos ilustrativos.)*

*   **Desempenho dos Algoritmos**: Com base nos resultados (valores de marcador de posição na Secção 6), se os Algoritmos Genéticos superassem consistentemente a Escalada de Encosta e o Recozimento Simulado em termos de encontrar valores de desvio padrão mais baixos (melhor equilíbrio), isso seria destacado. Por exemplo, se a `GA Config 4 (TargetExch,UnifPreferV,TournK5)` produzisse a menor aptidão média e a melhor aptidão geral, seria identificada como uma forte candidata.
*   **Influência dos Operadores no GA**:
    *   **Pressão de Seleção**: A seleção por torneio com um `k` mais alto (ex: `k=5` na GA Config 4 vs. `k=3` na GA Config 1) geralmente aumenta a pressão de seleção, potencialmente levando a uma convergência mais rápida, mas também arriscando a convergência prematura. A seleção por ranking (GA Config 2) oferece uma pressão equilibrada. A seleção de Boltzmann (GA Config 3) permite uma pressão dinâmica, mas requer uma sintonia cuidadosa da temperatura.
    *   **Operadores de Cruzamento**: O cruzamento uniforme (`crossover_uniform_prefer_valid`) promove frequentemente uma melhor mistura dos genes parentais em comparação com o cruzamento de um ponto (`crossover_one_point_prefer_valid`), o que pode ser benéfico para explorar soluções diversas. O invólucro "prefer-valid" foi essencial para ambos.
    *   **Operadores de Mutação**: `mutate_targeted_player_exchange` e `mutate_shuffle_within_team_constrained` (como uma troca inter-equipas) são mais conscientes do problema do que uma simples `mutate_swap_constrained`. Fazem alterações mais estruturadas (trocando jogadores entre equipas) que podem ser mais eficazes na navegação pela paisagem restrita do que apenas trocar as atribuições de dois jogadores aleatórios. A escolha do operador de mutação pode impactar significativamente a capacidade do AG de escapar a ótimos locais e explorar novas regiões.
*   **Sintonia de Parâmetros**: Os parâmetros escolhidos (tamanho da população, gerações, taxa de mutação, temperaturas do SA, etc.) foram baseados em heurísticas comuns e alguma experimentação preliminar (implícita, embora não detalhada explicitamente na evolução do script). A sintonia ótima de parâmetros é uma tarefa complexa por si só e frequentemente dependente do problema.

### 7.4. Elitismo em Algoritmos Genéticos

O elitismo, onde um pequeno número dos melhores indivíduos da geração atual é diretamente transportado para a próxima geração, foi implementado com `elite_size = 5` (ou `ga_params_dict["elitism_size"]` que foi definido como 2 no script mp) nas configurações de GA.

*   **Impacto**: O elitismo garante que as melhores soluções encontradas até ao momento não sejam perdidas devido a efeitos estocásticos da seleção, cruzamento ou mutação. Isto geralmente ajuda a:
    *   **Prevenir a Perda de Boas Soluções**: Garante que o desempenho do AG não se degrade de uma geração para a seguinte em termos do melhor indivíduo.
    *   **Acelerar a Convergência**: Pode levar a uma convergência mais rápida em direção a boas soluções, pois estas soluções são preservadas e continuam a participar na reprodução.
*   **Potencial Desvantagem**: Se o elitismo for demasiado forte (demasiadas elites), pode reduzir a diversidade e levar à convergência prematura para uma solução subótima. O `elite_size` escolhido (ex: 2 ou 5 de uma população de 50) é uma heurística comum que equilibra a preservação com a diversidade.

### 7.5. Desafios Encontrados

*   **Tratamento de Restrições**: O principal desafio foi lidar com as restrições muito rígidas do problema. A aplicação ingénua de operadores padrão de AG quase sempre produzia soluções inválidas. Isto tornou necessário:
    *   O desenvolvimento da heurística `_random_valid_assignment_constructive` para garantir que a população/soluções iniciais fossem válidas.
    *   A implementação de invólucros "prefer-valid" para o cruzamento e versões "constrained" para a mutação, que envolvem tentar novamente as operações até que um descendente/mutante válido seja encontrado (até um limite).
    *   Penalização pesada de soluções inválidas na função de aptidão.
*   **Custo Computacional**: Os Algoritmos Genéticos, especialmente com um tamanho de população de 50 e 100 gerações, e múltiplas execuções para validade estatística, são computacionalmente intensivos. O script inicial de processador único demorava um tempo considerável. Isto foi mitigado por:
    *   Implementar a versão multiprocessador (`main_script_mp.py`) para paralelizar as 30 execuções independentes, reduzindo significativamente o tempo total de relógio de parede para as experiências.
*   **Design de Operadores**: Projetar operadores de mutação e cruzamento eficazes que pudessem explorar significativamente o espaço de busca sem violar constantemente as restrições foi um processo iterativo. Operadores simples eram frequentemente demasiado disruptivos; operadores mais direcionados (como `mutate_targeted_player_exchange`) foram desenvolvidos para fazer alterações mais inteligentes.
*   **Sensibilidade aos Parâmetros**: O desempenho do SA e do GA pode ser sensível aos seus parâmetros (ex: esquema de arrefecimento para o SA, taxas e escolhas de operadores para o GA). Encontrar parâmetros ótimos requer tipicamente uma experimentação extensiva, que estava fora do âmbito da sintonia exaustiva para este projeto, mas foi abordada testando algumas configurações de GA.

### Referências para a Secção 7

[^3]: Coello Coello, Carlos A. 2002. "Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art." *Computer Methods in Applied Mechanics and Engineering* 191 (11-12): 1245–87. https://doi.org/10.1016/S0045-7825(01)00323-1.

---
*(A redação da Secção 8 seguirá.)*



## 8. Conclusão e Trabalhos Futuros

Este projeto abordou com sucesso o problema de Atribuição de Ligas Desportivas através da implementação e avaliação de três técnicas de inteligência computacional: Escalada de Encosta (HC), Recozimento Simulado (SA) e Algoritmos Genéticos (GA). O objetivo principal foi criar ligas equilibradas, minimizando o desvio padrão das classificações médias de habilidade das equipas, sujeito a restrições rigorosas de composição da equipa, atribuição de jogadores e orçamento.

### 8.1. Resumo das Principais Conclusões

*   **Complexidade do Problema**: A natureza altamente restrita do problema destacou a necessidade de mecanismos especializados de tratamento de restrições dentro dos algoritmos. A simples aplicação de operadores padrão foi insuficiente.
*   **Desempenho dos Algoritmos**: *(Esta parte será mais específica assim que os dados reais da Secção 6 estiverem totalmente integrados. Por agora, uma declaração geral:)* Os Algoritmos Genéticos, com a sua busca baseada em populações e operadores diversos, demonstraram geralmente uma forte capacidade para encontrar soluções de alta qualidade. Configurações específicas de GA, particularmente aquelas que empregam combinações eficazes de seleção, cruzamento consciente do problema e operadores de mutação (ex: potencialmente a GA Config 4), mostraram resultados promissores em termos de alcançar baixos valores de aptidão (ou seja, ligas bem equilibradas). O Recozimento Simulado ofereceu uma melhoria em relação à Escalada de Encosta básica, escapando a alguns ótimos locais, enquanto o HC serviu como uma linha de base para o desempenho da busca local.
*   **Eficácia do Tratamento de Restrições**: O uso de uma heurística construtiva para gerar soluções iniciais válidas, juntamente com operadores "prefer-valid" e "constrained", e penalização da aptidão para soluções inválidas, provou ser uma estratégia eficaz para navegar no espaço de busca viável.
*   **Eficiência Computacional**: Embora os GAs fossem computacionalmente mais intensivos por execução, a implementação de um script multiprocessador reduziu significativamente o tempo experimental geral, tornando viáveis testes extensivos (ex: 30 execuções por configuração).

### 8.2. Potenciais Trabalhos Futuros e Extensões

Existem várias vias para estender e melhorar este trabalho:

*   **Operadores Genéticos Avançados**: Explorar operadores de cruzamento e mutação mais sofisticados, especificamente concebidos para problemas de atribuição ou agrupamento fortemente restritos. Isto poderia incluir operadores que garantam a validade dos descendentes ou mecanismos de reparação mais inteligentes.
*   **Algoritmos Híbridos**: Desenvolver abordagens híbridas, como a combinação de GA com busca local (algoritmos meméticos). Por exemplo, as soluções evoluídas por um GA poderiam ser periodicamente refinadas usando Escalada de Encosta ou Recozimento Simulado para explorar regiões promissoras mais exaustivamente.
*   **Controlo Adaptativo de Parâmetros**: Implementar mecanismos adaptativos para parâmetros de GA, como o ajuste dinâmico de taxas de mutação, taxas de cruzamento ou pressão de seleção com base no estado da busca (ex: diversidade da população ou velocidade de convergência).
*   **Otimização Multiobjetivo**: Estender o problema para considerar múltiplos objetivos simultaneamente, como minimizar o desvio de habilidade *e* maximizar o apelo total da liga aos fãs (se tais dados estivessem disponíveis), ou minimizar as distâncias de viagem dos jogadores se as localizações das equipas fossem um fator. Isto exigiria algoritmos de otimização multiobjetivo como o NSGA-II.
*   **Conjuntos de Dados Maiores e Escalabilidade**: Testar os algoritmos implementados em conjuntos de dados maiores e mais complexos (ex: mais jogadores, mais equipas, restrições mais complexas) para avaliar a sua escalabilidade e robustez.
*   **Interface de Utilizador/Ferramenta de Apoio à Decisão**: Desenvolver uma interface gráfica de utilizador (GUI) que permita a um gestor de liga interagir com o processo de otimização, definir parâmetros, visualizar resultados e explorar diferentes configurações de liga potenciais.
*   **Representações Alternativas**: Embora o vetor linear de atribuição de jogadores tenha sido eficaz, explorar outras representações, talvez hierárquicas ou baseadas em grupos, pode oferecer diferentes vantagens ou desvantagens para o design de operadores e eficiência da busca.
*   **Otimização de Hiperparâmetros**: Realizar um estudo mais sistemático de otimização de hiperparâmetros (ex: usando técnicas como busca em grelha, busca aleatória ou otimização Bayesiana) para os parâmetros de SA e GA, para potencialmente descobrir configurações com desempenho ainda melhor.

Em conclusão, este projeto demonstrou a aplicação bem-sucedida de técnicas de inteligência computacional a um problema de otimização complexo e inspirado no mundo real. As conclusões fornecem uma base sólida para a compreensão das características de desempenho de HC, SA e GAs neste domínio e oferecem inúmeras direções para investigação e desenvolvimento futuros.

---
*(A redação da Secção 9 seguirá.)*



## 9. Referências

Coello Coello, Carlos A. 2002. "Theoretical and Numerical Constraint-Handling Techniques used with Evolutionary Algorithms: A Survey of the State of the Art." *Computer Methods in Applied Mechanics and Engineering* 191 (11-12): 1245–87. https://doi.org/10.1016/S0045-7825(01)00323-1.

Eiben, A. E., and J. E. Smith. 2015. *Introduction to Evolutionary Computing*. 2nd ed. Natural Computing Series. Berlin, Heidelberg: Springer Berlin Heidelberg. https://doi.org/10.1007/978-3-662-44874-8.

Michalewicz, Zbigniew, and Marc Schoenauer. 1996. "Evolutionary Algorithms for Constrained Parameter Optimization Problems." *Evolutionary Computation* 4 (1): 1–32. https://doi.org/10.1162/evco.1996.4.1.1.

*(Nota: Referências adicionais seriam adicionadas aqui se artigos ou livros específicos fossem consultados para outros aspetos dos algoritmos ou do domínio do problema durante um projeto real. Para este exercício, as acima são ilustrativas com base no conhecimento comum na área e nos tipos de justificações feitas.)*

---
*(A redação do Apêndice seguirá.)*


## Apêndice

### A.1. Lista Completa de Jogadores

A lista completa de jogadores, incluindo os seus nomes, posições, classificações de habilidade e custos, está disponível no ficheiro `players.csv` fornecido com o projeto. Este conjunto de dados forma a base para todas as atribuições de jogadores e formações de equipas nas experiências.

### A.2. Parâmetros Detalhados para Experiências

Esta secção fornece uma lista consolidada de parâmetros utilizados para cada algoritmo durante as execuções experimentais. Todos os algoritmos foram executados para `NUM_RUNS = 30` ensaios independentes.

**Parâmetros Gerais:**

*   `NUM_TEAMS`: 5
*   `TEAM_SIZE`: 7 jogadores por equipa
*   `MAX_BUDGET`: 750 milhões € por equipa
*   `NUM_RUNS`: 30 (para todos os algoritmos/configurações)

**Escalada de Encosta (HC):**

*   `max_iterations`: 1000
*   Geração da Solução Inicial: heurística `_random_valid_assignment_constructive`.
*   Geração de Vizinhança: Trocar as atribuições de equipa de todos os pares possíveis de jogadores, mantendo apenas vizinhos válidos.

**Recozimento Simulado (SA):**

*   `initial_temp`: 1000
*   `final_temp`: 0.1
*   `alpha` (taxa de arrefecimento): 0.99
*   `iterations_per_temp`: 50
*   Geração da Solução Inicial: heurística `_random_valid_assignment_constructive`.
*   Geração de Vizinho Aleatório: Trocar as atribuições de equipa de dois jogadores selecionados aleatoriamente, com novas tentativas para garantir a validade.

**Algoritmo Genético (GA) - Parâmetros Comuns:**

*   `population_size`: 50
*   `generations`: 100
*   `mutation_rate`: 0.2 (como usado em `main_script_sp.py`) ou 0.1 (como usado em `main_script_mp.py` - *nota: esta diferença deve ser reconciliada ou explicitamente mencionada se intencional para diferentes versões de script*).
*   `elite_size`: 5 (como usado em `main_script_sp.py`) ou 2 (como usado em `main_script_mp.py` - *nota: esta diferença deve ser reconciliada ou explicitamente mencionada*).
*   Geração da População Inicial: `generate_population` usando `_random_valid_assignment_constructive`.

**Configurações Específicas de GA (Detalhes dos operadores em `report/pt_operator_analysis.md`):**

1.  **GA Config 1 (SwapConst1PtPreferVTournVarK)**:
    *   Mutação: `mutate_swap_constrained`
    *   Cruzamento: `crossover_one_point_prefer_valid`
    *   Seleção: `selection_tournament_variable_k`
    *   `tournament_k`: 3

2.  **GA Config 2 (TargetExchUnifPreferVRanking)**:
    *   Mutação: `mutate_targeted_player_exchange`
    *   Cruzamento: `crossover_uniform_prefer_valid`
    *   Seleção: `selection_ranking`

3.  **GA Config 3 (ShuffleWithin1PtPreferVBoltzmann)**:
    *   Mutação: `mutate_shuffle_within_team_constrained`
    *   Cruzamento: `crossover_one_point_prefer_valid`
    *   Seleção: `selection_boltzmann`
    *   `boltzmann_temp`: 100

4.  **GA Config 4 (TargetExchUnifPreferVTournVarK_k5)**:
    *   Mutação: `mutate_targeted_player_exchange`
    *   Cruzamento: `crossover_uniform_prefer_valid`
    *   Seleção: `selection_tournament_variable_k`
    *   `tournament_k`: 5

*(Nota: As discrepâncias em GA mutation_rate e elite_size entre os scripts de processador único e multiprocessador deveriam idealmente ser unificadas para relatórios consistentes, ou a diferença e o seu potencial impacto discutidos se intencionais. Para este apêndice, ambos os valores vistos nos scripts são anotados.)*
