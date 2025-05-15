# Documento de Revisão de Código da Fase de Multiprocessamento (MP)

**Versão:** 1.0
**Data:** 15 de maio de 2025
**Revisor:** Manus (IA)

## 1. Introdução

Este documento detalha o processo e os resultados da revisão de código realizada para os scripts e módulos adaptados ou introduzidos durante a fase de multiprocessamento (MP) do projeto CIFO EXTENDED. O foco desta revisão foi avaliar a correção da implementação do paralelismo, a gestão de processos, a agregação de resultados e a manutenibilidade do código que lida com múltiplas execuções dos algoritmos.

Os principais ficheiros e aspetos revistos incluem:

*   Scripts principais de multiprocessamento (ex: `main_script_mp.py`, `main_script_mp_30_runs.py`, `main_script_mp_param_var.py`, `main_script_mp_final_param_var.py`): Orquestração das execuções paralelas.
*   Função(ões) de trabalhador (worker function): Encapsulamento da lógica de uma única execução de algoritmo para paralelização.
*   Utilização do módulo `multiprocessing` (ex: `Pool`, `map`, `apply_async`).
*   Mecanismos de recolha, agregação e armazenamento de resultados de múltiplas execuções (ex: escrita em ficheiros CSV de sumário).
*   Interação com os módulos da fase SP (`solution.py`, `evolution.py`, `operators.py`) no contexto multiprocesso.

## 2. Metodologia de Revisão

A revisão seguiu critérios semelhantes aos da fase SP, com ênfase adicional em aspetos específicos do multiprocessamento:

*   **Correção da Paralelização:** O paralelismo está implementado corretamente? As tarefas são distribuídas e os resultados recolhidos de forma eficaz?
*   **Gestão de Processos:** O `Pool` de processos é gerido adequadamente (criação, utilização, fecho)?
*   **Isolamento e Partilha de Dados:** Os processos trabalhadores operam de forma isolada quando necessário? A partilha de dados (se houver) é segura?
*   **Tratamento de Erros em Contexto Paralelo:** Como são tratados os erros que ocorrem num processo trabalhador?
*   **Eficiência do Multiprocessamento:** O overhead do multiprocessamento é justificado pelos ganhos de tempo? Existem estrangulamentos na distribuição ou recolha de tarefas/resultados?
*   **Clareza do Código de Orquestração:** Os scripts principais são fáceis de entender no que diz respeito à lógica de paralelização?
*   **Robustez da Agregação de Resultados:** Os resultados de todas as execuções são corretamente agregados e guardados?

## 3. Observações Gerais e Pontos Positivos

*   **Reutilização Eficaz dos Módulos SP:** Os módulos `solution.py`, `evolution.py`, e `operators.py` foram reutilizados com sucesso no contexto multiprocesso, minimizando a reescrita de código.
*   **Abstração com `multiprocessing.Pool`:** A utilização de `Pool` simplificou significativamente a gestão dos processos trabalhadores e a distribuição de tarefas.
*   **Função Trabalhadora Bem Definida:** A criação de uma função trabalhadora (worker function) para encapsular cada execução individual de algoritmo foi uma boa prática, facilitando a paralelização com `pool.map()`.
*   **Flexibilidade dos Scripts Principais MP:** A criação de diferentes scripts principais para diferentes fases de experimentação (5 execuções, 30 execuções, variação de parâmetros) permitiu uma abordagem organizada e adaptada a cada necessidade de análise.
*   **Saída Estruturada em CSV:** A padronização da saída dos resultados agregados em ficheiros CSV facilitou enormemente a análise subsequente e a geração de relatórios e gráficos.

## 4. Pontos de Melhoria Identificados e Ações Tomadas (Histórico)

Durante o desenvolvimento da fase MP, alguns desafios e pontos de melhoria foram abordados:

*   **Gestão da Aleatoriedade em Processos Paralelos:**
    *   **Observação:** Garantir que cada processo trabalhador utiliza uma semente de aleatoriedade diferente (ou que a aleatoriedade é gerida de forma a não produzir resultados idênticos devido a sementes partilhadas inadvertidamente) foi uma consideração inicial.
    *   **Ação:** Embora o `multiprocessing` geralmente isole bem os processos, foi verificado que a instanciação de objetos `random` ou a chamada a `random.seed()` dentro da função trabalhadora (se necessário para reprodutibilidade específica de uma execução) não causava conflitos diretos, pois cada processo tem o seu próprio espaço de memória. Para reprodutibilidade global de um conjunto de execuções, a estratégia de sementes teria de ser gerida no script principal antes de lançar os workers, se fosse um requisito estrito.
*   **Overhead de Serialização/Desserialização:**
    *   **Observação:** A passagem de objetos complexos (como instâncias de `LeagueSolution` ou grandes `player_data`) para os processos trabalhadores e o retorno de resultados podem incorrer em overhead de serialização (pickling/unpickling).
    *   **Ação:** Para `player_data`, que é grande mas imutável durante as execuções, esta é carregada uma vez no processo pai e implicitamente herdada (ou copiada eficientemente em sistemas POSIX com copy-on-write) pelos processos filhos. Para os parâmetros de configuração e os resultados, o volume de dados era geralmente pequeno, minimizando este overhead. Otimizações mais agressivas (ex: usando `multiprocessing.shared_memory` para `player_data`) foram consideradas mas não implementadas devido à complexidade adicional versus o ganho percebido no contexto do projeto.
*   **Debugging de Processos Paralelos:**
    *   **Observação:** Depurar código que corre em múltiplos processos pode ser mais desafiador.
    *   **Ação:** Utilização de `print` statements estratégicos dentro da função trabalhadora durante o desenvolvimento. Teste da lógica da função trabalhadora de forma sequencial primeiro, antes de a executar em paralelo. Ferramentas de debugging mais avançadas não foram necessárias dada a estrutura relativamente simples da paralelização.
*   **Tratamento de Exceções nos Trabalhadores:**
    *   **Observação:** Uma exceção não tratada num processo trabalhador pode fazer com que o processo termine silenciosamente ou que o `Pool` se comporte de forma inesperada.
    *   **Ação:** Envolver a lógica principal da função trabalhadora num bloco `try...except` para capturar exceções, registá-las (ex: imprimir ou retornar uma mensagem de erro como parte do resultado) e garantir que o processo trabalhador termina de forma controlada. Isto foi progressivamente melhorado nos scripts.
*   **Fecho Correto do `Pool`:**
    *   **Observação:** É crucial fechar (`pool.close()`) e juntar (`pool.join()`) o `Pool` para libertar recursos e garantir que todos os processos terminaram.
    *   **Ação:** Todos os scripts principais MP implementam corretamente o padrão `pool.close()` seguido de `pool.join()` após a submissão de todas as tarefas.
*   **Número de Processos no Pool:**
    *   **Observação:** Inicialmente, o número de processos era fixo. Seria mais flexível usar `os.cpu_count()`.
    *   **Ação:** Os scripts foram atualizados para usar `os.cpu_count()` como padrão para o tamanho do `Pool`, permitindo uma melhor utilização dos recursos da máquina onde o código é executado, com a possibilidade de configurar manualmente se necessário.

## 5. Revisão Específica dos Scripts MP

*   **`main_script_mp.py` (e variantes):**
    *   **Pontos Fortes:** Estrutura clara para definir configurações, preparar argumentos para a função trabalhadora, executar o `Pool` e processar os resultados. A lógica de agregação (cálculo de médias, desvios padrão, identificação do melhor) é correta.
    *   **Melhorias (Históricas/Consideradas):** Refatoração para reduzir a duplicação de código entre os diferentes scripts MP (ex: `main_script_mp.py`, `main_script_mp_30_runs.py`). Parte desta refatoração ocorreu ao longo do projeto, centralizando funções comuns de preparação de argumentos ou processamento de resultados. Uma maior abstração poderia ser alcançada com uma classe ou funções de utilidade mais genéricas para a orquestração de experiências.
*   **Função(ões) Trabalhadora(s):**
    *   **Pontos Fortes:** Encapsula bem uma única execução. A lógica de timing e retorno de resultados é clara.
    *   **Melhorias (Históricas/Consideradas):** Melhoria no tratamento de exceções, como mencionado. Garantir que todos os dados necessários são passados como argumentos para manter o isolamento.

## 6. Conclusões da Revisão de Código MP

O código da fase de multiprocessamento (MP) demonstra uma implementação funcional e eficaz da paralelização para as necessidades do projeto CIFO EXTENDED. A utilização do módulo `multiprocessing` foi bem-sucedida em acelerar a experimentação e permitir a recolha de dados estatisticamente significativos.

Os principais desafios do multiprocessamento, como a gestão de aleatoriedade, overhead e debugging, foram considerados e abordados a um nível satisfatório para os objetivos do projeto. As melhorias implementadas ao longo do desenvolvimento, como o tratamento de exceções nos trabalhadores e a gestão dinâmica do tamanho do `Pool`, aumentaram a robustez e a eficiência do código.

A estrutura adotada, com scripts principais dedicados a diferentes conjuntos de experiências e uma função trabalhadora clara, provou ser uma abordagem flexível e manutenível.

