"""Sistema de Detecção de Fraudes em Transações Financeiras com Machine Learning.

Pacote do TCC (IFRS – Campus Canoas, ADS). Organização:

- ``config``      : sementes, caminhos e parâmetros globais do experimento.
- ``data``        : carga das bases (ULB Credit Card, PaySim) e base sintética de teste.
- ``features``    : pré-processamento e engenharia de atributos.
- ``models``      : fábrica de modelos e estratégias de balanceamento.
- ``evaluation``  : métricas, função de custo, escolha de limiar e latência.
- ``experiments`` : protocolo experimental (CV estratificada + busca aleatória).
- ``explain``     : explicabilidade (SHAP e LIME).
- ``drift``       : análise exploratória de concept drift.
- ``artifact``    : treinamento e serialização do modelo final para a API.
"""

__version__ = "1.0.0"
