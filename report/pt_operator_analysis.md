## Análise Aprofundada dos Operadores de Algoritmos Genéticos para o Problema de Atribuição de Ligas Desportivas

Este documento fornece uma análise abrangente dos operadores de Algoritmos Genéticos (AG) — seleção, cruzamento e mutação — conforme implementados e utilizados no CIFO_EXTENDED_Project para resolver o altamente constrangido Problema de Atribuição de Ligas Desportivas. Compara estas implementações com operadores padrão frequentemente encontrados em exemplos académicos (como os do repositório de turma `inesmcm26/CIFO-24-25`) e justifica as escolhas de design feitas no projeto, referenciando práticas estabelecidas em computação evolutiva.

### 1. Contexto: O Problema de Atribuição de Ligas Desportivas e a Sua Representação

Uma compreensão fundamental do problema e da sua representação de solução escolhida é primordial antes de dissecar os operadores. O Problema de Atribuição de Ligas Desportivas requer a atribuição de um número fixo de jogadores (35) a um número fixo de equipas (5). Cada equipa deve aderir a constrangimentos rigorosos:

*   **Tamanho da Equipa:** Exatamente 7 jogadores por equipa.
*   **Equilíbrio Posicional:** Cada equipa deve ter 1 Guarda-Redes (GR), 2 Defesas (DEF), 2 Médios (MED) e 2 Avançados (AVA).
*   **Restrição Orçamental:** O custo total dos jogadores em cada equipa não deve exceder um orçamento predefinido (ex: 750 milhões).

O projeto emprega um **vetor de atribuição de jogadores (codificação linear)** como representação da sua solução. Esta é uma lista onde o índice corresponde a um ID de jogador, e o valor nesse índice é o ID da equipa à qual o jogador está atribuído (ex: `self.assignment[player_idx] = team_id`). Esta representação é detalhada em `solution.py` e é a pedra angular sobre a qual todos os operadores de AG atuam.

### 2. Operadores de Seleção: Escolher Progenitores para Reprodução

Os operadores de seleção determinam quais indivíduos da população atual são escolhidos para se tornarem progenitores para a próxima geração. O objetivo é favorecer indivíduos mais aptos, mantendo a diversidade. O projeto implementa vários mecanismos de seleção padrão, detalhados em `operators.py`.

#### 2.1. Operadores de Seleção Implementados no Projeto

*   **Seleção por Torneio (`selection_tournament_variable_k`)**: Este método envolve a seleção aleatória de `k` indivíduos da população e a escolha do melhor (valor de aptidão mais baixo no nosso problema de minimização) entre eles para ser um progenitor. Este é um método amplamente utilizado devido à sua eficiência e pressão de seleção ajustável através do tamanho do torneio `k`. (Eiben e Smith 2015, 87).[^1]
    *   *Implementação no Projeto*: O nosso `selection_tournament_variable_k(population, players_data, k)` identifica corretamente o indivíduo mais apto de um torneio aleatório. O argumento `players_data` é crucial para a avaliação da aptidão, que é tratada apropriadamente.

*   **Seleção por Ranking (`selection_ranking`)**: Os indivíduos são primeiro ordenados com base na sua aptidão. Em seguida, a cada indivíduo é atribuída uma probabilidade de seleção proporcional ao seu ranking, em vez da sua aptidão bruta. Isto pode evitar a convergência prematura quando alguns indivíduos superaptos dominam a população. (Eiben e Smith 2015, 88).[^1]
    *   *Implementação no Projeto*: `selection_ranking(population, players_data)` ordena a população por aptidão, atribui rankings e calcula as probabilidades de seleção em conformidade. Esta é uma implementação robusta, adequada para manter a pressão de seleção sem ser excessivamente distorcida por valores de aptidão atípicos.

*   **Seleção de Boltzmann (`selection_boltzmann`)**: Este método, inspirado no recozimento simulado (simulated annealing), ajusta as probabilidades de seleção com base na aptidão e num parâmetro semelhante à temperatura. Temperaturas mais altas levam a probabilidades de seleção mais uniformes (exploração), enquanto temperaturas mais baixas aumentam a pressão de seleção em direção a indivíduos mais aptos (explotação). (Eiben e Smith 2015, 90).[^1]
    *   *Implementação no Projeto*: `selection_boltzmann(population, players_data, temperature, k=1)` implementa isto, tratando potenciais problemas numéricos como probabilidades `inf` ou `NaN`. O parâmetro de temperatura permite um controlo dinâmico sobre a pressão de seleção, o que pode ser benéfico ao longo da execução evolutiva.

#### 2.2. Comparação com Exemplos da Turma e Justificação

Os exemplos da turma (`inesmcm26/CIFO-24-25/src/ga/selection.py`) apresentam principalmente a **Seleção Proporcional à Aptidão (Fitness Proportionate Selection - FPS)**, também conhecida como Seleção por Roleta (Roulette Wheel Selection). Embora a FPS seja um método clássico, tem problemas conhecidos:

*   **Convergência Prematura:** Indivíduos altamente aptos podem dominar rapidamente a população, especialmente em problemas de minimização com grandes variações nos valores de aptidão, levando a uma perda de diversidade.
*   **Estagnação:** Se os valores de aptidão forem muito próximos, a pressão de seleção torna-se fraca.
*   **Problemas de Escala:** Requer que os valores de aptidão sejam positivos; para minimização, a aptidão muitas vezes precisa ser invertida e escalada, o que pode ser complicado.

**Justificação para as Escolhas do Projeto:**

*   A escolha do projeto de implementar a Seleção por Torneio, Ranking e Boltzmann em detrimento da FPS direta é bem justificada. Estes métodos geralmente oferecem um melhor controlo sobre a pressão de seleção e são menos suscetíveis aos problemas de escala e convergência prematura frequentemente associados à FPS, particularmente em problemas de otimização complexos e constrangidos como o nosso.
*   Os operadores implementados são padrão e comprovadamente eficazes numa vasta gama de aplicações de AG.

### 3. Operadores de Cruzamento: Combinar Material Genético dos Progenitores

Os operadores de cruzamento (ou recombinação) combinam material genético de duas soluções progenitoras para criar um ou mais descendentes, com o objetivo de herdar características benéficas de ambos os progenitores.

#### 3.1. Operadores de Cruzamento Implementados no Projeto

O ficheiro `operators.py` do projeto incluía inicialmente operadores básicos de cruzamento de um ponto e uniforme. No entanto, devido aos constrangimentos apertados do problema, estes operadores ingénuos quase invariavelmente produzem descendentes inválidos (violando o tamanho da equipa, o equilíbrio posicional ou o orçamento). A inovação crucial foi o desenvolvimento de versões "prefer-valid":

*   **Cruzamento de Um Ponto com Preferência por Válidos (`crossover_one_point_prefer_valid`)**: Este operador realiza um cruzamento padrão de um ponto nas listas `assignment` de dois progenitores. Um único ponto de cruzamento é escolhido e os segmentos são trocados. Crucialmente, após criar um descendente, a sua validade é verificada usando `child_solution.is_valid(players_data)`. Se o descendente for inválido, o processo de cruzamento (potencialmente com novos pontos de cruzamento aleatórios ou progenitores, se assim concebido, embora a implementação atual tente novamente com os mesmos progenitores mas novo ponto) é tentado até `max_attempts`. Se um descendente válido não for encontrado, o último descendente gerado (que pode ser inválido) ou uma cópia de um progenitor é devolvido. (Eiben e Smith 2015, 50, discutem o cruzamento geral; a validade específica do problema é fundamental aqui).[^1]

*   **Cruzamento Uniforme com Preferência por Válidos (`crossover_uniform_prefer_valid`)**: Para cada gene (atribuição de jogador) no descendente, este operador escolhe aleatoriamente qual progenitor contribuirá com esse gene. Semelhante à versão de um ponto, incorpora uma verificação de validade e um mecanismo de repetição (`max_attempts`) para favorecer a geração de soluções válidas. (Eiben e Smith 2015, 51).[^1]

#### 3.2. Comparação com Exemplos da Turma e Justificação

Os exemplos da turma (`inesmcm26/CIFO-24-25/src/ga/crossover.py`) fornecem `standard_crossover` (um ponto) e `uniform_crossover`. Estes são adequados para problemas não constrangidos ou problemas com representações simples (ex: strings binárias).

**Justificação para as Escolhas do Projeto:**

*   **Inadequação Direta dos Operadores Padrão:** Aplicar diretamente os operadores de cruzamento padrão da biblioteca da turma ao nosso vetor de atribuição de jogadores seria ineficaz. A probabilidade de um descendente criado pela troca aleatória de segmentos de duas soluções progenitoras válidas também ser válido é extremamente baixa devido à complexa interação das regras de composição da equipa.
*   **Necessidade de Tratamento de Constrangimentos:** A estratégia "prefer-valid" é uma abordagem pragmática e comum para lidar com constrangimentos em AGs. Tenta usar mecanismos de operadores padrão, mas filtra ou repete para validade. Isto é muitas vezes mais simples de implementar do que conceber operadores de cruzamento altamente complexos e específicos do problema que *garantam* a validade desde o início.
*   **Cruzamentos Específicos para Permutação (ex: OX1, PMX):** Operadores de cruzamento como o Cruzamento por Ordem (OX1) ou o Cruzamento Parcialmente Mapeado (PMX), frequentemente vistos em exemplos da turma para problemas como o Problema do Caixeiro Viajante (TSP), **não são aplicáveis** aqui. A nossa representação é uma atribuição direta, não uma permutação de jogadores ou equipas.
*   **Melhorias Potenciais (Fora do Âmbito Atual):** Estratégias mais avançadas poderiam envolver mecanismos de reparação (corrigir ativamente um descendente inválido) ou conceber cruzamentos que operam em estruturas de nível superior (ex: trocar equipas válidas inteiras ou grupos de jogadores de forma equilibrada). No entanto, isto adiciona uma complexidade significativa. A abordagem "prefer-valid" estabelece um equilíbrio.

### 4. Operadores de Mutação: Introduzir Variação Genética

Os operadores de mutação introduzem pequenas alterações aleatórias no material genético de um indivíduo, ajudando a manter a diversidade na população e a evitar a estagnação prematura, permitindo que o AG explore novas áreas do espaço de busca.

#### 4.1. Operadores de Mutação Implementados no Projeto

Semelhante ao cruzamento, os operadores de mutação iniciais não constrangidos revelaram-se problemáticos. O projeto evoluiu para usar versões constrangidas que priorizam a validade da solução, conforme detalhado em `operators.py`:

*   **Mutação por Troca Constrangida (`mutate_swap_constrained`)**: Este operador seleciona aleatoriamente dois jogadores e troca as suas atribuições de equipa. Após a troca, a validade da solução é verificada. Se a mutação resultar numa solução inválida, o processo é repetido até `max_attempts`. Se um mutante válido não for encontrado, a solução original é devolvida inalterada. Isto garante que a mutação não degrade a população com indivíduos inválidos se uma mutação válida for difícil de encontrar rapidamente. (Semelhante à mutação por troca geral, mas com verificação de validade crítica; Eiben e Smith 2015, 54).[^1]

*   **Mutação por Troca de Jogadores Direcionada (`mutate_targeted_player_exchange`)**: Este é um operador mais consciente do problema. Seleciona aleatoriamente duas equipas diferentes e depois tenta trocar um jogador escolhido aleatoriamente da primeira equipa com um jogador escolhido aleatoriamente da segunda equipa. A validade é verificada e são realizadas repetições. Se nenhuma troca válida for encontrada após `max_attempts`, a solução original é devolvida. Este operador tem maior probabilidade de fazer alterações significativas, tendo ao mesmo tempo uma melhor hipótese de preservar (ou encontrar rapidamente) a validade em comparação com uma reatribuição aleatória de um único jogador.

*   **Mutação por Baralhamento Dentro da Equipa Constrangida (`mutate_shuffle_within_team_constrained`)**: Apesar do nome, a implementação parece selecionar um jogador de uma equipa escolhida e tenta trocá-lo com um jogador de *outra* equipa (semelhante a `mutate_targeted_player_exchange`, mas talvez com uma ênfase de seleção diferente para o jogador/equipa inicial). Também emprega verificações de validade e repetições. O nome pode sugerir um baralhamento intra-equipa, mas o código indica uma troca inter-equipas. Se fosse um verdadeiro baralhamento intra-equipa (ex: reatribuir jogadores *já na mesma equipa* a diferentes papéis dentro dessa equipa, se os papéis fizessem parte do cromossoma, ou apenas baralhar a sua ordem se isso importasse), seria um tipo diferente de operador. Dada a implementação atual, atua como outra forma de troca de jogadores inter-equipas.

#### 4.2. Comparação com Exemplos da Turma e Justificação

Os exemplos da turma (`inesmcm26/CIFO-24-25/src/ga/mutation.py`) incluem `binary_standard_mutation` (inversão de bits para strings/listas binárias) e `swap_mutation` (troca genérica de genes).

**Justificação para as Escolhas do Projeto:**

*   **Inaplicabilidade da Mutação Binária:** `binary_standard_mutation` é irrelevante, pois a nossa representação é baseada em inteiros (IDs de equipa).
*   **Limitações da Troca Genérica:** Uma `swap_mutation` genérica (trocar os IDs de equipa de dois jogadores) é a base para `mutate_swap_constrained`. No entanto, sem o mecanismo de verificação de constrangimentos e repetição, invalidaria frequentemente as soluções.
*   **Operadores Específicos do Problema:** `mutate_targeted_player_exchange` e `mutate_shuffle_within_team_constrained` (como uma troca inter-equipas) são mais sofisticados. Tentam alterações que são estruturalmente mais complexas do que a reatribuição aleatória de equipa de um único jogador, permitindo potencialmente uma exploração mais eficaz do espaço de busca válido.
*   **Mutação por Inversão:** `inversion_mutation`, tipicamente usada para problemas baseados em permutação, não é aplicável aqui.
*   **Mutação por Deslizamento (Creep Mutation):** `creep_mutation` (adicionar pequenos valores) é para representações numéricas, não para as nossas atribuições categóricas de equipa.

### 5. Estratégia Geral para Design de Operadores e Tratamento de Constrangimentos

O design dos operadores de AG do projeto reflete uma estratégia comum e eficaz para enfrentar problemas de otimização altamente constrangidos:

1.  **Escolha da Representação:** Uma representação linear e direta (vetor `assignment`) foi escolhida pela sua simplicidade e compatibilidade com muitos mecanismos de operadores.
2.  **Verificação de Validade:** Um método robusto `is_valid()` (`LeagueSolution.is_valid()`) é central para todo o processo. Atua como o árbitro da viabilidade da solução.
3.  **Penalização da Aptidão:** Soluções inválidas são fortemente penalizadas na função de aptidão (`LeagueSolution.fitness()`), removendo-as efetivamente da competição durante a seleção.
4.  **Adaptação de Operadores:** Conceitos padrão de cruzamento e mutação são adaptados para:
    *   **Preferir Validade:** Tentar a operação e repetir se uma solução inválida for produzida (ex: `crossover_one_point_prefer_valid`, `mutate_swap_constrained`).
    *   **Lógica Consciente do Problema:** Conceber operadores que compreendam inerentemente alguns aspetos da estrutura do problema para fazer alterações mais inteligentes (ex: `mutate_targeted_player_exchange`).
5.  **Heurísticas Construtivas para Inicialização:** O método `_random_valid_assignment_constructive` garante que a população inicial comece com indivíduos válidos, fornecendo uma boa base para o processo evolutivo.

Esta abordagem multifacetada é geralmente preferível a tentar conceber operadores que *sempre* produzam descendentes válidos, pois estes últimos podem tornar-se excessivamente complexos e podem restringir indevidamente a busca. (Michalewicz e Schoenauer 1996).[^2]

### 6. Conclusão e Recomendações

Os operadores de AG implementados no CIFO_EXTENDED_Project são bem adequados para o Problema de Atribuição de Ligas Desportivas, particularmente devido às suas adaptações para lidar com constrangimentos complexos. Os mecanismos de seleção são padrão e robustos. Os operadores de cruzamento e mutação, especialmente as versões "prefer-valid" e "constrained", são essenciais para navegar eficazmente no espaço de busca.

**Principais Pontos Fortes:**

*   **Consciência dos Constrangimentos:** A principal força reside no tratamento explícito da validade da solução dentro ou imediatamente após a aplicação do operador.
*   **Fundações Padrão:** Os operadores baseiam-se em princípios de AG bem compreendidos, adaptados para o problema específico.
*   **Variedade:** A disponibilidade de múltiplos tipos de operadores de seleção, cruzamento e mutação permite flexibilidade e experimentação na configuração do AG (como visto no dicionário `ga_configs`).

**Áreas Potenciais para Exploração Futura (Fora do Âmbito Atual):**

*   **Probabilidades Adaptativas de Operadores:** Implementar mecanismos para adaptar as probabilidades de aplicação de diferentes operadores de cruzamento ou mutação durante a execução, com base no seu sucesso passado na geração de soluções mais aptas ou válidas.
*   **Operadores de Reparação Mais Sofisticados:** Se as estratégias "prefer-valid" falharem frequentemente em encontrar soluções válidas rapidamente, poderiam ser desenvolvidos operadores de reparação dedicados para pegar num descendente inválido e tentar alterá-lo minimamente para se tornar válido.
*   **Análise do Desempenho dos Operadores:** Analisar sistematicamente quais operadores (e os seus parâmetros, como `max_attempts`) contribuem mais eficazmente para encontrar boas soluções para este problema específico.

Em conclusão, o conjunto de operadores neste projeto demonstra uma abordagem ponderada para aplicar AGs a uma tarefa de otimização desafiadora e constrangida. As escolhas de design priorizam a descoberta de soluções válidas e de alta qualidade, integrando a lógica de tratamento de constrangimentos diretamente com mecanismos de operadores estabelecidos.

### Referências

[^1]: Eiben, A. E., e J. E. Smith. 2015. *Introduction to Evolutionary Computing*. 2ª ed. Natural Computing Series. Berlim, Heidelberg: Springer Berlin Heidelberg.

[^2]: Michalewicz, Zbigniew, e Marc Schoenauer. 1996. "Evolutionary Algorithms for Constrained Parameter Optimization Problems." *Evolutionary Computation* 4 (1): 1–32. https://doi.org/10.1162/evco.1996.4.1.1.

---
*Esta análise baseia-se no código e documentação do CIFO_EXTENDED_Project, especificamente `operators.py`, `solution.py` e os principais scripts de execução.*
