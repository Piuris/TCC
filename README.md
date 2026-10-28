# Sistema de Detecção de Fraudes em Transações Financeiras com Machine Learning

Implementação do TCC de **João Vitor Pires** — Tecnologia em Análise e Desenvolvimento de
Sistemas, IFRS Campus Canoas (orientador: Prof. Dr. Rafael Coimbra Pinto).

Análise comparativa de algoritmos de classificação e de detecção de anomalias em bases
desbalanceadas, com tratamento do desbalanceamento, função de custo, explicabilidade
(SHAP/LIME), análise de *concept drift* e um protótipo de API REST de predição.

## Estrutura

```
src/fraude/            pacote principal (python -m fraude ...)
  config.py            sementes, caminhos, cenários de custo, parâmetros do experimento
  data.py              carga da base ULB, do PaySim e da base sintética de teste
  features.py          engenharia de atributos e pré-processamento (sem vazamento)
  models.py            algoritmos × técnicas de balanceamento, espaços de busca
  evaluation.py        métricas, função de custo, escolha de limiar, latência, bootstrap
  experiments.py       protocolo experimental (CV estratificada + busca aleatória)
  ensemble.py          ensembles por votação suave das melhores configurações
  stats.py             CV repetida, Friedman/Nemenyi, t corrigido, seleção por parcimônia
  calibration.py       calibração de probabilidades (Platt/isotônica) e confiabilidade
  reporting.py         tabelas e figuras de resultados
  explain.py           SHAP (global/local) e LIME
  drift.py             validação temporal e concept drift (estático × retreinado)
  artifact.py          treinamento e serialização do modelo servido
  eda.py               análise exploratória
  consolidate.py       tabelas e figuras numeradas de todas as bases para o texto
src/fraude_api/        API FastAPI (main.py, schemas.py)
tests/                 testes automatizados (pytest)
scripts/               reproduce.ps1 (fluxo completo), load_test.py (carga da API)
notebooks/             01_analise_exploratoria.ipynb, 02_resultados.ipynb
data/                  bases (não versionadas)
artifacts/             modelos serializados (.joblib + .json de metadados)
reports/<base>/        resultados: eda/, experiments/, explainability/, drift/
```

## Instalação

Requer Python 3.12.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest                       # 89 testes
```

## Bases de dados

| Base | Comando | Observação |
|---|---|---|
| ULB Credit Card (principal) | `python -m fraude download --dataset creditcard` | Baixada do OpenML (id 1597), sem necessidade de conta. 284.807 transações, 492 fraudes. |
| PaySim (complementar) | `python -m fraude download --dataset paysim` | Requer CLI do Kaggle configurada, ou baixe [ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) e salve como `data/paysim.csv`. |
| Sintética (testes) | `python -m fraude download --dataset synthetic` | Mesmo esquema da ULB; apenas para validar o código. |

No PaySim são mantidas somente as operações `TRANSFER` e `CASH_OUT` (únicas com fraude no
simulador) e é descartado `isFlaggedFraud` (saída de uma regra do simulador = vazamento de rótulo).
A base tem 6,3 milhões de linhas; use `--sample-frac 0.2` (amostra estratificada) para
execuções viáveis em máquina pessoal.

**Atalho do simulador.** No PaySim o agente fraudador transfere sempre o saldo inteiro da
conta: `amount == oldbalanceOrg` em 97,8% das fraudes e em 0,0% das transações legítimas.
Com os saldos da origem, todos os algoritmos, inclusive a regressão logística, chegam a
AUC-PR ≈ 1,0, e a comparação deixa de ser informativa. Por isso há dois cenários:

| Cenário | `--dataset` | Atributos |
|---|---|---|
| Completo | `paysim` | todos (comparável à literatura; modelo servido pela API) |
| Sem saldo da origem | `paysim_sem_saldo` | valor, hora, tipo e conta de destino; sem `oldbalanceOrg`/`newbalanceOrig` nem nada derivado deles |

Fluxo completo dos dois cenários: `bash scripts/run_paysim.sh`.

## Execução dos experimentos

```powershell
python -m fraude eda --dataset creditcard
python -m fraude experiment --dataset creditcard --n-iter 10      # 21 configurações (~2h30)
python -m fraude experiment --dataset creditcard --models xgboost lightgbm `
    --n-iter 50 --out-name experiments_busca50                    # busca ampliada (~2h)
python -m fraude ensemble --dataset creditcard                    # ensembles das melhores
python -m fraude compare --dataset creditcard --repeats 3         # testes estatísticos + seleção
python -m fraude train-final --dataset creditcard                 # modelo selecionado, calibrado
python -m fraude explain --dataset creditcard
python -m fraude drift --dataset creditcard --windows 6
python -m fraude drift --dataset creditcard --label-delay 1       # simula atraso dos rótulos
python -m fraude consolidate                                      # reports/consolidado/
```

Ou tudo de uma vez: `powershell -ExecutionPolicy Bypass -File scripts\reproduce.ps1`.

Opções úteis de `experiment`: `--models lightgbm xgboost`, `--balancing none smote`,
`--n-iter`, `--cv`, `--sample-frac`, `--quick` (n_iter=3, k=3) e `--no-resume`. Os
resultados são gravados a cada configuração concluída; uma execução interrompida
continua de onde parou.

### Protocolo experimental

| Aspecto | Escolha |
|---|---|
| Algoritmos supervisionados | Regressão Logística (referência), Random Forest, XGBoost, LightGBM, MLP |
| Não supervisionado | Isolation Forest (rótulos usados só para avaliar/ajustar hiperparâmetros) |
| Balanceamento | sem tratamento, SMOTE, subamostragem aleatória, pesos de classe (cost-sensitive) |
| Duplicatas | removidas **antes** da separação (atributos + rótulo iguais, ignorando o instante): sem isso, cópias caíam no treino e no teste |
| Separação | 80% treino / 20% teste, estratificada, semente 42; teste usado uma única vez |
| Ajuste | busca aleatória, CV estratificada k=5, seleção por AUC-PR; busca ampliada (n_iter=50) para XGBoost e LightGBM |
| Ensembles | média das probabilidades das melhores configurações de XGBoost + LightGBM (+ Random Forest) |
| Vazamento | pré-processamento e reamostragem dentro do `imblearn.Pipeline` → reajustados só no treino de cada partição |
| Limiar | escolhido nas predições *out-of-fold* do treino: F1 máximo e custo mínimo em 4 cenários |
| Métricas | precisão, revocação, F1, MCC, AUC-ROC, AUC-PR (IC 95% bootstrap), Brier, matriz de confusão, custo |
| Custo computacional | tempo de treino, latência por transação (média, p50, p95, p99), vazão em lote |
| Comparação estatística | CV repetida 5×3 (mesmas partições para todos); Friedman + Nemenyi; teste t corrigido de Nadeau-Bengio contra a melhor, com ajuste de Holm |
| Seleção do modelo final | parcimônia com margem prática: entre as configurações não significativamente piores que a melhor **e** a no máximo 0,01 de AUC-PR dela, a de maior AUC-PR entre as mais rápidas (latências até 0,5 ms acima da mínima contam como empate) |
| Calibração | Platt, isotônica ou nenhuma, escolhida pelo Brier em CV interna sobre as predições out-of-fold; o limiar é escolhido já na escala calibrada |

Cenários de custo (`config.py`): conservador FN:FP = 10:1, **moderado 20:1** (cenário
principal da tabela), agressivo 100:1 e **valor_transacao**, em que deixar passar uma
fraude custa o valor dela (piso de 25) e bloquear uma legítima custa 5.

Decisões de projeto que valem ser discutidas no texto:

- **`Time` da ULB não é usado cru**: é a posição da transação na base e não generaliza;
  vira hora do dia com codificação cíclica (seno/cosseno).
- **Winsorização só em atributos monetários**: nas componentes PCA, valores extremos são
  frequentemente o próprio sinal de fraude.
- **MLP sensível a custo**: o `MLPClassifier` não aceita `sample_weight`; usa-se a
  reamostragem proporcional ao custo (Zadrozny; Langford; Abe, 2003), equivalente em
  esperança à perda ponderada.
- **Subamostragem**: a proporção é hiperparâmetro; o relatório compara o Brier antes e
  depois da correção de Dal Pozzolo et al. (2015), aplicada também no artefato servido.

### Saídas (`reports/<base>/`)

- `eda/` — estatísticas (`eda.json`, `descriptive_stats.csv`) e figuras.
- `experiments/` — `results.csv` (tabela comparativa), `results.json` (tudo, incluindo
  hiperparâmetros e notas de cada partição), `summary.md`, `figures/` (heatmap AUC-PR,
  curvas PR/ROC, custo × limiar, matrizes de confusão, desempenho × latência).
- `experiments_busca50/`, `experiments_ensemble/` — mesmas saídas, para a busca ampliada
  e para os ensembles.
- `comparacao/` — notas da CV repetida (`repeated_cv_scores.csv`), testes
  (`statistical_tests.json`), tabela (`comparison.csv`, `summary.md`), ranking com
  diferença crítica (`ranking_cd.png`) e a configuração escolhida (`selection.json`).
- `calibracao/` — método escolhido, Brier/ECE/log loss antes e depois, e diagrama de
  confiabilidade.
- `explainability/` — importância global SHAP, beeswarm, waterfall e LIME para um
  verdadeiro positivo, um falso positivo e um falso negativo, e a concordância SHAP×LIME.
- `drift/` — AUC-PR por janela temporal (estático × janela deslizante × acumulado),
  IC bootstrap, PSI do escore e do valor.

`reports/consolidado/` (comando `consolidate`) reúne todas as bases para o capítulo de
resultados: `resultados.md` com tabelas e figuras numeradas (título acima, fonte abaixo) e os
pontos de discussão com os números preenchidos; `tabelas/*.csv` (`;` e vírgula decimal, abre
direto no Excel) e `figuras/*.png`. Os números vêm dos arquivos de resultado: depois de
retreinar, basta rodar o comando de novo.

## API de predição

```powershell
uvicorn fraude_api.main:app --port 8000
```

Documentação interativa: <http://localhost:8000/docs>.

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | estado e modelos carregados |
| GET | `/v1/models` | metadados: configuração, limiar, métricas de teste, latência, versões |
| POST | `/v1/creditcard/predict` | uma transação ULB (`Time`, `V1`…`V28`, `Amount`) |
| POST | `/v1/creditcard/predict/batch` | até 1.000 transações |
| POST | `/v1/paysim/predict` | uma transação PaySim |
| POST | `/v1/paysim/predict/batch` | até 1.000 transações |

Resposta:

```json
{
  "transaction_id": "tx-001",
  "fraud_probability": 0.9731,
  "threshold": 0.2140,
  "decision": "bloquear",
  "is_fraud": true,
  "top_factors": [
    {"feature": "V14", "value": -7.21, "contribution": 3.12, "direction": "aumenta_risco"}
  ],
  "explanation_method": "shap_tree",
  "model_version": "creditcard-lightgbm-class_weight-2026-10-17T14:02:11",
  "latency_ms": 4.8
}
```

`?explain=false` desliga o cálculo dos fatores. O limiar pode ser sobrescrito com
`FRAUDE_THRESHOLD_CREDITCARD` / `FRAUDE_THRESHOLD_PAYSIM`. Entradas inválidas (campo
ausente, tipo errado, valor negativo, NaN, campo desconhecido) retornam HTTP 422.

Teste de carga (com a API rodando):

```powershell
python scripts/load_test.py --requests 2000 --concurrency 16
python scripts/load_test.py --requests 2000 --concurrency 16 --no-explain
```

## Testes

`pytest` cobre: engenharia de atributos e ausência de vazamento no pré-processamento;
métricas, função de custo (comparada a força bruta), limiar e correção de calibração;
treino e predição de todas as 21 combinações algoritmo × balanceamento; reamostragem só
no `fit`; protocolo experimental de ponta a ponta com retomada; aditividade do SHAP;
drift com atraso de rótulos; remoção de duplicatas sem vazamento entre treino e teste;
calibração (melhora do Brier e preservação da ordem); teste t corrigido, Holm, Friedman e a
regra de parcimônia (incluindo a margem prática); ensemble igual à média dos membros; custo por valor; e testes de
contrato da API.

## Reprodutibilidade

Versões fixadas em `requirements.txt`; semente global 42 (`--seed` para alterar);
`reports/environment.txt` registra o `pip freeze` da execução; cada artefato guarda as
versões das bibliotecas, os hiperparâmetros e a data de treino.

## Limitações conhecidas

- Atributos da ULB anonimizados por PCA: a explicabilidade nessa base aponta componentes
  sem significado de negócio — daí o uso complementar do PaySim.
- A ULB cobre apenas dois dias de setembro de 2013, com cartões europeus; a análise de
  drift tem poucas fraudes por janela (ver os intervalos de confiança).
- No PaySim, as inconsistências de saldo são em parte artefato do simulador, o que
  infla o desempenho.
- Os rótulos são tratados como disponíveis imediatamente, exceto no experimento com
  `--label-delay`.
- A ULB não tem identificador do cartão: não é possível criar atributos de comportamento
  (frequência de uso, desvio do valor habitual), que são a principal alavanca de
  desempenho em sistemas reais.
- Com dois dias de dados, a importância da hora do dia no SHAP pode refletir um padrão
  real (mais fraude de madrugada) ou uma coincidência da amostra.
- `reports/creditcard_v1_com_duplicatas/` guarda a primeira execução, feita antes da remoção
  das duplicatas, para quantificar o efeito do vazamento.
