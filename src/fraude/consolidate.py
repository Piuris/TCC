"""Consolida os resultados de todas as bases para o capítulo de resultados.

Lê o que os demais comandos gravaram em ``reports/`` e ``artifacts/`` e gera
``reports/consolidado/``:

- ``resultados.md``: tabelas e figuras numeradas, com título acima e fonte
  abaixo (padrão ABNT), e os pontos de discussão com os números já preenchidos;
- ``tabelas/tabela_NN_*.csv``: os dados de cada tabela (separador ``;`` e
  vírgula decimal, para abrir direto no Excel/Word em português);
- ``figuras/figura_NN_*.png``: cópias numeradas das figuras citadas.

Nenhum número é digitado à mão: retreinar e rodar de novo atualiza o documento.
Partes cujos resultados não existem são omitidas.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import ensemble  # noqa: F401
from .config import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR
from .evaluation import savings_vs_best_trivial
from .models import BALANCING_LABELS, MODEL_LABELS
from .stats import LATENCY_TOL_MS, PRACTICAL_MARGIN, select_parsimonious

log = logging.getLogger(__name__)

DATASETS = {
    "creditcard": "ULB (cartão de crédito)",
    "paysim_sem_saldo": "PaySim sem saldo da origem",
    "paysim": "PaySim completo",
}
ARTIFACT_KIND = {"creditcard": "creditcard", "paysim": "paysim", "paysim_sem_saldo": "paysim_restrito"}
SCENARIOS = {
    "conservador": "Conservador (FN 100, FP 10)",
    "moderado": "Moderado (FN 100, FP 5)",
    "agressivo": "Agressivo (FN 100, FP 1)",
    "valor_transacao": "Valor da transação (FN = valor, piso 25; FP 5)",
}
STRATEGIES = {"estatico": "Estático", "janela_deslizante": "Janela deslizante", "acumulado": "Acumulado"}
SOURCES = {"experiments": "", "busca50": " [busca n=50]", "ensemble": ""}
FONTE = "Fonte: elaborado pelo autor."


def br(x, nd: int = 3) -> str:
    """Número com vírgula decimal; vazio para ausentes."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}".replace(",", ".")
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def pct(x, nd: int = 1) -> str:
    return br(x, nd) + "%"


def label(model: str, balancing: str, source: str = "experiments") -> str:
    lab = MODEL_LABELS.get(model, model)
    if balancing != "misto":
        lab += " + " + BALANCING_LABELS.get(balancing, balancing)
    return lab + SOURCES.get(source, f" [{source}]")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _records(reports: Path, ds: str, sub: str = "experiments") -> list[dict]:
    return _read_json(reports / ds / sub / "results.json") or []


def _md_table(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


class Document:
    """Acumula o texto e numera tabelas e figuras."""

    def __init__(self, out: Path):
        self.out = out
        self.lines: list[str] = []
        self.n_tab = 0
        self.n_fig = 0
        for d in ("tabelas", "figuras"):
            shutil.rmtree(out / d, ignore_errors=True)
            (out / d).mkdir(parents=True, exist_ok=True)

    def add(self, *lines: str) -> None:
        self.lines += [*lines, ""]

    def table(self, shown: pd.DataFrame, title: str, name: str, data: pd.DataFrame | None = None,
              note: str = "") -> str:
        """``shown`` vai formatado para o texto; ``data`` (numérico) vai para o CSV."""
        self.n_tab += 1
        ref = f"Tabela {self.n_tab}"
        (data if data is not None else shown).to_csv(
            self.out / "tabelas" / f"tabela_{self.n_tab:02d}_{name}.csv",
            sep=";", decimal=",", index=False, encoding="utf-8-sig")
        self.add(f"**{ref}** – {title}", _md_table(shown), (note + " " if note else "") + FONTE)
        return ref

    def figure(self, src: Path, title: str, name: str, note: str = "") -> str | None:
        if not src.exists():
            return None
        self.n_fig += 1
        ref = f"Figura {self.n_fig}"
        dest = f"figura_{self.n_fig:02d}_{name}.png"
        shutil.copyfile(src, self.out / "figuras" / dest)
        self.add(f"**{ref}** – {title}", f"![{ref}](figuras/{dest})", (note + " " if note else "") + FONTE)
        return ref


def duplicate_leakage(reports: Path, data_dir: Path = DATA_DIR) -> dict | None:
    """Quantas fraudes do teste da execução sem deduplicação tinham cópia exata no treino."""
    split = reports / "creditcard_v1_com_duplicatas" / "experiments" / "split.npz"
    if not split.exists() or not (data_dir / "creditcard.csv").exists():
        return None
    from .data import TIME_COLUMNS, load_creditcard

    ds = load_creditcard(data_dir / "creditcard.csv")
    sp = np.load(split)
    cols = [c for c in ds.X.columns if c not in TIME_COLUMNS]
    keys = pd.util.hash_pandas_object(ds.X[cols].assign(_y=ds.y.to_numpy()), index=False).to_numpy()
    y = ds.y.to_numpy()
    te_fraud = sp["test"][y[sp["test"]] == 1]
    in_train = np.isin(keys[te_fraud], keys[sp["train"]])
    return {"test_frauds": int(len(te_fraud)), "with_copy_in_train": int(in_train.sum())}


def paysim_shortcut(reports: Path, data_dir: Path = DATA_DIR) -> dict | None:
    """Mede o atalho do simulador na base inteira (TRANSFER e CASH_OUT); guarda em cache."""
    cache = reports / "paysim" / "eda" / "atalho_simulador.json"
    if cache.exists():
        return _read_json(cache)
    if not (data_dir / "paysim.csv").exists():
        return None
    from .data import load_paysim

    ds = load_paysim(data_dir / "paysim.csv")
    X, fraud = ds.X, ds.y.to_numpy() == 1
    amount, old_o, new_o = X["amount"], X["oldbalanceOrg"], X["newbalanceOrig"]
    rules = {
        "valor igual ao saldo da origem": np.isclose(amount, old_o),
        "saldo da origem zerado após a transação": ((old_o > 0) & (new_o == 0)).to_numpy(),
        "saldo da origem bate exatamente (antes − valor = depois)": np.isclose(old_o - amount, new_o),
    }
    out = {"n": int(len(fraud)), "n_fraud": int(fraud.sum()), "rules": {
        k: {"fraude_pct": float(100 * v[fraud].mean()), "legitima_pct": float(100 * v[~fraud].mean())}
        for k, v in rules.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    return out


def selections(comparison: pd.DataFrame) -> dict[str, str]:
    """Configuração escolhida pela regra original e pela regra com margem prática."""
    table = comparison.set_index("config")
    latency = table["latency_mean_ms"].to_dict()
    return {
        "original": select_parsimonious(table, latency, margin=np.inf, latency_tol_ms=0.0),
        "revisada": select_parsimonious(table, latency),
        "melhor": table["auc_pr_mean"].idxmax(),
    }


def _key_label(key: str) -> str:
    cfg, _, source = key.partition("@")
    model, _, balancing = cfg.partition("__")
    return label(model, balancing, source or "experiments")


def section_bases(doc: Document, reports: Path) -> None:
    rows = []
    for ds, name in DATASETS.items():
        info = _read_json(reports / ds / "experiments" / "dataset.json")
        if not info:
            continue
        frac = info["name"].partition("[")[2].rstrip("]") or "100%"
        rows.append({"Base": name, "Amostra": frac, "Transações": info["n_rows"],
                     "Fraudes": info["n_fraud"], "Taxa de fraude (%)": info["fraud_rate_pct"],
                     "Atributos de entrada": info["n_features"],
                     "Duplicatas removidas": info.get("duplicates_removed", 0),
                     "Fraudes duplicadas removidas": info.get("duplicate_frauds_removed", 0)})
    if not rows:
        return
    data = pd.DataFrame(rows)
    shown = data.assign(**{c: data[c].map(br) for c in data.columns if data[c].dtype.kind in "iu"},
                        **{"Taxa de fraude (%)": data["Taxa de fraude (%)"].map(lambda v: br(v, 4))})
    doc.add("## Bases de dados")
    doc.table(shown, "Bases utilizadas, após a remoção de duplicatas", "bases", data,
              note="No PaySim, só operações TRANSFER e CASH_OUT, em amostra estratificada.")


def section_duplicates(doc: Document, reports: Path, facts: dict, data_dir: Path) -> None:
    v1 = {(r["model"], r["balancing"]): r for r in _records(reports, "creditcard_v1_com_duplicatas")}
    v2 = {(r["model"], r["balancing"]): r for r in _records(reports, "creditcard")}
    common = [k for k in v2 if k in v1]
    if not common:
        return
    rows = []
    for k in common:
        a, b = v1[k], v2[k]
        rows.append({"Configuração": label(*k),
                     "AUC-PR CV (com dup.)": a["cv_auc_pr_mean"], "AUC-PR CV (sem dup.)": b["cv_auc_pr_mean"],
                     "AUC-PR teste (com dup.)": a["test_metrics"]["default"]["auc_pr"],
                     "AUC-PR teste (sem dup.)": b["test_metrics"]["default"]["auc_pr"]})
    data = pd.DataFrame(rows)
    data["Diferença no teste"] = data["AUC-PR teste (sem dup.)"] - data["AUC-PR teste (com dup.)"]
    data = data.sort_values("AUC-PR teste (sem dup.)", ascending=False)
    shown = data.assign(**{c: data[c].map(br) for c in data.columns[1:]})
    leak = duplicate_leakage(reports, data_dir)
    facts["dup_mean_delta_test"] = float(data["Diferença no teste"].mean())
    facts["dup_mean_delta_cv"] = float((data["AUC-PR CV (sem dup.)"] - data["AUC-PR CV (com dup.)"]).mean())
    facts["dup_leak"] = leak
    doc.add("## Efeito das transações duplicadas (ULB)")
    if leak:
        doc.add(f"Na execução sem deduplicação, {leak['with_copy_in_train']} das {leak['test_frauds']} "
                "fraudes do conjunto de teste tinham uma cópia exata (mesmos atributos e rótulo, "
                "ignorando o instante) no treino.")
    doc.table(shown, "AUC-PR com e sem a remoção das duplicatas, por configuração (ULB)",
              "duplicatas", data,
              note=f"Diferença média no teste: {br(facts['dup_mean_delta_test'])}; "
                   f"na validação cruzada: {br(facts['dup_mean_delta_cv'])}.")


def _comparison_rows(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        m = r["test_metrics"]["cost_moderado"]
        lo, hi = r["test_auc_pr_ci95"]
        rows.append({"Algoritmo": MODEL_LABELS[r["model"]], "Balanceamento": BALANCING_LABELS[r["balancing"]],
                     "AUC-PR CV": r["cv_auc_pr_mean"], "DP CV": r["cv_auc_pr_std"],
                     "AUC-PR teste": m["auc_pr"], "IC95 inf": lo, "IC95 sup": hi, "AUC-ROC": m["auc_roc"],
                     "Precisão": m["precision"], "Revocação": m["recall"], "F1": m["f1"], "MCC": m["mcc"],
                     "Latência (ms)": r["latency_mean_ms"]})
    return pd.DataFrame(rows).sort_values("AUC-PR teste", ascending=False)


def section_comparison(doc: Document, reports: Path, ds: str, facts: dict) -> None:
    records = _records(reports, ds)
    if not records:
        return
    data = _comparison_rows(records)
    shown = pd.DataFrame({
        "Algoritmo": data["Algoritmo"], "Balanceamento": data["Balanceamento"],
        "AUC-PR CV": [f"{br(a)} ± {br(b)}" for a, b in zip(data["AUC-PR CV"], data["DP CV"])],
        "AUC-PR teste [IC 95%]": [f"{br(a)} [{br(b)}; {br(c)}]"
                                  for a, b, c in zip(data["AUC-PR teste"], data["IC95 inf"], data["IC95 sup"])],
        **{c: data[c].map(br) for c in ["AUC-ROC", "Precisão", "Revocação", "F1", "MCC"]},
        "Latência (ms)": data["Latência (ms)"].map(lambda v: br(v, 2)),
    })
    supervised = data[~data["Algoritmo"].str.startswith("Isolation")]
    iforest = data[data["Algoritmo"].str.startswith("Isolation")]
    facts[f"{ds}_test_range"] = (float(supervised["AUC-PR teste"].min()), float(supervised["AUC-PR teste"].max()))
    facts[f"{ds}_logreg"] = float(data.loc[data["Algoritmo"].str.startswith("Regressão"), "AUC-PR teste"].max())
    if len(iforest):
        facts[f"{ds}_iforest"] = (float(iforest["AUC-PR teste"].iloc[0]), float(iforest["AUC-ROC"].iloc[0]))
    doc.add(f"## Comparação dos algoritmos — {DATASETS[ds]}")
    doc.table(shown, f"Desempenho no conjunto de teste — {DATASETS[ds]}", f"comparacao_{ds}", data,
              note="Precisão, revocação, F1 e MCC no limiar de custo mínimo do cenário moderado "
                   "(FN:FP = 20:1), escolhido nas predições out-of-fold do treino. "
                   "CV: média ± desvio-padrão em 5 partições.")
    exp = reports / ds / "experiments" / "figures"
    doc.figure(exp / "auc_pr_heatmap.png", f"AUC-PR no teste por algoritmo e balanceamento — {DATASETS[ds]}",
               f"heatmap_{ds}")
    if ds == "creditcard":
        doc.figure(exp / "pr_curves.png", "Curvas precisão-revocação no teste — ULB", "curvas_pr_ulb")
        doc.figure(exp / "tradeoff_latency_aucpr.png", "AUC-PR × latência por transação — ULB",
                   "latencia_ulb")


def section_statistics(doc: Document, reports: Path, facts: dict) -> None:
    rows, done = [], []
    for ds in ("creditcard", "paysim_sem_saldo"):
        cmp_path = reports / ds / "comparacao" / "comparison.csv"
        tests = _read_json(reports / ds / "comparacao" / "statistical_tests.json")
        if not cmp_path.exists() or not tests:
            continue
        comp = pd.read_csv(cmp_path)
        t = comp.set_index("config")
        sel = selections(comp)
        fr = tests["friedman_nemenyi"]
        facts[f"{ds}_selection"] = {k: {"label": _key_label(v), "auc": float(t.loc[v, "auc_pr_mean"]),
                                        "lat": float(t.loc[v, "latency_mean_ms"]),
                                        "test": float(t.loc[v, "test_auc_pr"])} for k, v in sel.items()}
        facts[f"{ds}_n_equiv"] = (int(t["equivalent_to_best"].sum()), len(t))
        for role, name in [("melhor", "Maior média"), ("original", "Regra original"),
                           ("revisada", "Regra com margem prática")]:
            k = sel[role]
            rows.append({"Base": DATASETS[ds], "Critério": name, "Configuração": _key_label(k),
                         "AUC-PR CV repetida": t.loc[k, "auc_pr_mean"], "p (Holm) vs. melhor": t.loc[k, "p_holm"],
                         "Latência (ms)": t.loc[k, "latency_mean_ms"], "AUC-PR teste": t.loc[k, "test_auc_pr"]})
        done.append((ds, fr, tests, int(t["equivalent_to_best"].sum()), len(t)))
    if not rows:
        return
    doc.add("## Comparação estatística e escolha do modelo final",
            "CV repetida 5×3 no treino (15 partições, as mesmas para todas as configurações), "
            "teste de Friedman com pós-teste de Nemenyi e teste t corrigido de Nadeau-Bengio contra a "
            "melhor configuração, com ajuste de Holm (α = 0,05).")
    summary = pd.DataFrame([{
        "Base": DATASETS[ds], "Configurações": n, "χ² de Friedman": fr["statistic"], "p": fr["p_value"],
        "Diferença crítica (Nemenyi)": fr["critical_difference"],
        "Não significativamente piores que a melhor": eq} for ds, fr, tests, eq, n in done])
    doc.table(summary.assign(**{"χ² de Friedman": summary["χ² de Friedman"].map(lambda v: br(v, 2)),
                                "p": summary["p"].map(lambda v: f"{v:.1e}".replace(".", ",")),
                                "Diferença crítica (Nemenyi)": summary["Diferença crítica (Nemenyi)"].map(
                                    lambda v: br(v, 2))}),
              "Testes de comparação múltipla", "friedman", summary)
    data = pd.DataFrame(rows)
    shown = data.assign(**{c: data[c].map(br) for c in ["AUC-PR CV repetida", "p (Holm) vs. melhor",
                                                        "AUC-PR teste"]},
                        **{"Latência (ms)": data["Latência (ms)"].map(lambda v: br(v, 2))})
    doc.table(shown, "Configuração escolhida por cada critério", "selecao", data,
              note="Regra original: a de menor latência entre as não significativamente piores que a "
                   f"melhor. Regra com margem prática: além disso, até {br(PRACTICAL_MARGIN, 2)} de AUC-PR "
                   f"abaixo da melhor; latências até {br(LATENCY_TOL_MS, 1)} ms acima da mínima contam "
                   "como empate, e vence a de maior AUC-PR.")
    for ds, *_ in done:
        doc.figure(reports / ds / "comparacao" / "ranking_cd.png",
                   f"Ranking médio na CV repetida e diferença crítica de Nemenyi — {DATASETS[ds]}",
                   f"ranking_{ds}")


def _artifact(ds: str, artifacts: Path) -> dict | None:
    return _read_json(artifacts / f"model_{ARTIFACT_KIND[ds]}.json")


def section_final(doc: Document, reports: Path, facts: dict, artifacts: Path) -> None:
    rows, cal_rows, cost_rows = [], [], []
    for ds, name in DATASETS.items():
        m = _artifact(ds, artifacts)
        if not m:
            continue
        t, lat = m["test_metrics"], m["latency"]
        cal = m.get("calibration") or {}
        rows.append({"Base": name, "Configuração": label(m["model"], m["balancing"]),
                     "Calibração": {"sigmoid": "Platt", "isotonic": "Isotônica"}.get(cal.get("method"), "Nenhuma"),
                     "Limiar": t["threshold"], "AUC-PR": t["auc_pr"], "AUC-ROC": t["auc_roc"],
                     "Precisão": t["precision"], "Revocação": t["recall"], "F1": t["f1"], "MCC": t["mcc"],
                     "VP": t["tp"], "FP": t["fp"], "FN": t["fn"],
                     "Latência média (ms)": lat["latency_mean_ms"], "Latência p99 (ms)": lat["latency_p99_ms"]})
        if cal.get("test_raw"):
            raw, calib = cal["test_raw"], cal["test_calibrated"]
            cal_rows.append({"Base": name, "Método": rows[-1]["Calibração"],
                             "Brier antes": raw["brier"], "Brier depois": calib["brier"],
                             "ECE antes": raw["ece"], "ECE depois": calib["ece"],
                             "Log loss antes": raw["log_loss"], "Log loss depois": calib["log_loss"],
                             "AUC-PR antes": raw["auc_pr"], "AUC-PR depois": calib["auc_pr"]})
        for sc, sc_name in SCENARIOS.items():
            c = m["test_cost_all_scenarios"].get(sc)
            if c:
                cost_rows.append({"Base": name, "Cenário": sc_name, "Limiar": c["threshold"], "Custo": c["cost"],
                                  "Aprovar tudo": c["cost_approve_all"], "Bloquear tudo": c["cost_block_all"],
                                  "Economia vs. aprovar tudo (%)": c["savings_vs_approve_all_pct"],
                                  "Economia vs. melhor trivial (%)": savings_vs_best_trivial(c)})
        facts[f"{ds}_final"] = rows[-1]
    if not rows:
        return
    doc.add("## Modelos finais")
    data = pd.DataFrame(rows)
    nd = {"Limiar": 4, "Latência média (ms)": 2, "Latência p99 (ms)": 2}
    shown = data.assign(**{c: data[c].map(lambda v, c=c: br(v, nd.get(c, 3)))
                           for c in data.columns if c not in ("Base", "Configuração", "Calibração")})
    doc.table(shown, "Desempenho dos modelos finais no conjunto de teste", "modelos_finais", data,
              note="Limiar de custo mínimo do cenário moderado, na escala calibrada. "
                   "VP/FP/FN: verdadeiros positivos, falsos positivos e falsos negativos.")

    if cal_rows:
        cdata = pd.DataFrame(cal_rows)
        facts["calibration"] = cdata
        cshown = cdata.assign(**{c: cdata[c].map(lambda v: br(v, 5)) for c in cdata.columns[2:8]},
                              **{c: cdata[c].map(br) for c in cdata.columns[8:]})
        doc.add("## Calibração das probabilidades",
                "Método escolhido pelo menor Brier em validação cruzada interna sobre as predições "
                "out-of-fold do treino; valores medidos no teste.")
        doc.table(cshown, "Métricas de calibração no teste, antes e depois da calibração",
                  "calibracao", cdata, note="ECE: erro de calibração esperado (10 faixas).")
        doc.figure(reports / "creditcard" / "calibracao" / "reliability.png",
                   "Diagrama de confiabilidade do modelo final — ULB", "confiabilidade_ulb")

    if cost_rows:
        kdata = pd.DataFrame(cost_rows)
        facts["costs"] = kdata
        kshown = kdata.assign(Limiar=kdata["Limiar"].map(lambda v: br(v, 6)),
                              **{c: kdata[c].map(lambda v: br(v, 2)) for c in ["Custo", "Aprovar tudo",
                                                                            "Bloquear tudo"]},
                              **{c: kdata[c].map(lambda v: br(v, 1)) for c in kdata.columns[-2:]})
        doc.add("## Função de custo")
        doc.table(kshown, "Custo do modelo final nos quatro cenários, comparado às políticas triviais",
                  "custos", kdata,
                  note="Melhor trivial: a mais barata entre aprovar e bloquear todas as transações.")
        doc.figure(reports / "creditcard" / "experiments" / "figures" / "cost_curves_moderado.png",
                   "Custo em função do limiar, cenário moderado — ULB", "custo_ulb")


def section_explain(doc: Document, reports: Path, facts: dict) -> None:
    cols, overlaps = {}, []
    for ds, name in DATASETS.items():
        e = _read_json(reports / ds / "explainability" / "explainability.json")
        if not e:
            continue
        top = e["global_top10"][:5]
        cols[name] = [f"{t['feature']} ({br(t['mean_abs_shap'], 2)})" for t in top]
        facts[f"{ds}_shap"] = [t["feature"] for t in top]
        for case, c in e["cases"].items():
            overlaps.append({"Base": name, "Caso": case.replace("_", " "),
                             "Probabilidade": c["probability"], "Sobreposição SHAP × LIME (top 5)": c["top5_overlap"]})
    if not cols:
        return
    doc.add("## Explicabilidade")
    data = pd.DataFrame({"Posição": range(1, 6), **cols})
    doc.table(data, "Cinco atributos mais importantes (média do |SHAP|) no modelo final de cada base",
              "shap_top5", note="Atributos no espaço após a engenharia de atributos.")
    if overlaps:
        odata = pd.DataFrame(overlaps)
        doc.table(odata.assign(Probabilidade=odata["Probabilidade"].map(lambda v: br(v, 4))),
                  "Concordância entre SHAP e LIME nas explicações locais", "shap_lime", odata,
                  note="Número de atributos em comum entre os cinco mais relevantes de cada método.")
    for ds, name in DATASETS.items():
        doc.figure(reports / ds / "explainability" / "shap_beeswarm.png",
                   f"Distribuição dos valores SHAP por atributo — {name}", f"shap_{ds}")
    doc.figure(reports / "creditcard" / "explainability" / "shap_waterfall_verdadeiro_positivo.png",
               "Explicação local (SHAP) de uma fraude detectada — ULB", "shap_local_ulb")


def section_drift(doc: Document, reports: Path, facts: dict) -> None:
    rows = []
    for ds, name in DATASETS.items():
        for sub, delay in (("drift", 0), ("drift_atraso1", 1)):
            p = reports / ds / sub / "drift.csv"
            if not p.exists():
                continue
            d = pd.read_csv(p)
            mean = d.groupby("strategy")["auc_pr"].mean()
            frauds = d.groupby("window")["n_fraud"].first()
            rows.append({"Base": name, "Atraso dos rótulos (janelas)": delay,
                         **{STRATEGIES[s]: mean.get(s, np.nan) for s in STRATEGIES},
                         "Pior janela (estático)": d.loc[d.strategy == "estatico", "auc_pr"].min(),
                         "PSI máximo do escore": d["psi_score"].max(),
                         "Fraudes por janela": f"{frauds.min()}–{frauds.max()}"})
    if not rows:
        return
    data = pd.DataFrame(rows)
    facts["drift"] = data
    num = [*STRATEGIES.values(), "Pior janela (estático)", "PSI máximo do escore"]
    doc.add("## Concept drift",
            "Treino nos primeiros 40% da linha do tempo e teste em 6 janelas consecutivas. Estático: "
            "nunca retreinado; janela deslizante: retreinado com a janela anterior; acumulado: "
            "retreinado com todo o histórico disponível.")
    doc.table(data.assign(**{c: data[c].map(br) for c in num}),
              "AUC-PR média nas janelas temporais, por estratégia de retreino", "drift", data,
              note="PSI acima de 0,25 indica mudança relevante de distribuição.")
    for ds, name in DATASETS.items():
        if ds != "paysim":
            doc.figure(reports / ds / "drift" / "drift.png", f"AUC-PR por janela temporal — {name}",
                       f"drift_{ds}")


def section_shortcut(doc: Document, reports: Path, facts: dict, data_dir: Path) -> None:
    sc = paysim_shortcut(reports, data_dir)
    if not sc:
        return
    facts["shortcut"] = sc
    data = pd.DataFrame([{"Regra": k, "Fraudes (%)": v["fraude_pct"], "Legítimas (%)": v["legitima_pct"]}
                         for k, v in sc["rules"].items()])
    doc.add("## O atalho do simulador PaySim")
    doc.table(data.assign(**{c: data[c].map(lambda v: br(v, 1)) for c in data.columns[1:]}),
              "Frequência de padrões de saldo em fraudes e legítimas (PaySim, TRANSFER e CASH_OUT)",
              "atalho_paysim", data,
              note=f"Base inteira: {br(sc['n'])} transações, {br(sc['n_fraud'])} fraudes.")


def section_discussion(doc: Document, facts: dict) -> None:
    pts = []
    if "dup_mean_delta_test" in facts:
        leak = facts.get("dup_leak")
        pts.append("**Vazamento por duplicatas.** "
                   + (f"{leak['with_copy_in_train']} das {leak['test_frauds']} fraudes do teste tinham cópia "
                      "exata no treino. " if leak else "")
                   + f"Removê-las mudou a AUC-PR no teste em {br(facts['dup_mean_delta_test'])} em média, "
                   f"contra {br(facts['dup_mean_delta_cv'])} na CV: o vazamento inflava sobretudo a "
                   "estimativa no teste.")
    sel = facts.get("creditcard_selection")
    if sel:
        o, r, b = sel["original"], sel["revisada"], sel["melhor"]
        n_eq, n = facts["creditcard_n_equiv"]
        pts.append("**Poder do teste estatístico e regra de seleção.** Com 15 partições, "
                   f"{n_eq} de {n} configurações não diferem significativamente da melhor na ULB. A regra "
                   f"original escolheria {o['label']} (AUC-PR {br(o['auc'])}, {br(o['lat'], 2)} ms), "
                   f"{br(b['auc'] - o['auc'])} abaixo da melhor para economizar "
                   f"{br(r['lat'] - o['lat'], 2)} ms. A margem prática levou a {r['label']} "
                   f"(AUC-PR {br(r['auc'])}, {br(r['lat'], 2)} ms). A regra foi revista depois de ver os "
                   "resultados, o que deve ser declarado no texto.")
    if "shortcut" in facts:
        rule = next(iter(facts["shortcut"]["rules"].items()))
        rng = facts.get("paysim_test_range")
        rng2 = facts.get("paysim_sem_saldo_test_range")
        pts.append("**Atalho do PaySim.** A regra \"" + rule[0] + f"\" vale para {pct(rule[1]['fraude_pct'])} "
                   f"das fraudes e {pct(rule[1]['legitima_pct'])} das legítimas. "
                   + (f"Com todos os atributos, a AUC-PR de teste dos modelos supervisionados fica entre "
                      f"{br(rng[0])} e {br(rng[1])} "
                      f"(regressão logística: {br(facts['paysim_logreg'])}); " if rng else "")
                   + (f"sem os saldos da origem, entre {br(rng2[0])} e {br(rng2[1])}." if rng2 else ""))
    costs = facts.get("costs")
    if costs is not None:
        v = costs[costs["Cenário"].str.startswith("Valor")]
        worst = v.loc[(v["Economia vs. aprovar tudo (%)"] - v["Economia vs. melhor trivial (%)"]).idxmax()]
        pts.append("**Economia enganosa.** No cenário de custo pelo valor, " + worst["Base"] + " mostra "
                   f"{pct(worst['Economia vs. aprovar tudo (%)'])} de economia contra aprovar tudo, mas "
                   f"{pct(worst['Economia vs. melhor trivial (%)'])} contra a melhor política trivial: com "
                   "fraudes de centenas de milhares e FP fixo em 5, o limiar ótimo cai para perto de zero. "
                   "Os custos de erro precisam estar na escala dos valores da base.")
    cal = facts.get("calibration")
    if cal is not None:
        worse = cal[cal["Brier depois"] > cal["Brier antes"]]
        better = cal[cal["Brier depois"] <= cal["Brier antes"]]
        txt = "**Calibração.** "
        if len(better):
            txt += "Reduziu o Brier em " + ", ".join(
                f"{r.Base} ({br(r._3, 5)} → {br(r._4, 5)})" for r in better.itertuples()) + ". "
        if len(worse):
            txt += ("Piorou no teste em " + ", ".join(
                f"{r.Base} ({r.Método}: {br(r._3, 5)} → {br(r._4, 5)})" for r in worse.itertuples())
                    + ": a escolha feita no treino nem sempre se confirma no teste.")
        pts.append(txt)
    drift = facts.get("drift")
    if drift is not None:
        d0 = drift[drift["Atraso dos rótulos (janelas)"] == 0]
        txt = "; ".join(f"{r.Base}: estático {br(r.Estático)}, acumulado {br(r.Acumulado)}, PSI máx. "
                        f"{br(r._7)}" for r in d0.itertuples())
        pts.append("**Drift.** AUC-PR média por estratégia — " + txt + ". O retreino trouxe ganho "
                   "pequeno na ULB e nenhum no PaySim, cujo simulador gera fraudes pelo mesmo processo o "
                   "tempo todo (PSI baixo). Poucas fraudes por janela deixam os intervalos de confiança "
                   "largos, e a ULB cobre só dois dias.")
    if "creditcard_iforest" in facts:
        ap, roc = facts["creditcard_iforest"]
        pts.append(f"**Abordagem não supervisionada.** O Isolation Forest ficou com AUC-PR {br(ap)} na "
                   f"ULB, muito abaixo dos supervisionados, embora sua AUC-ROC seja {br(roc)}: em bases "
                   "desbalanceadas a ROC esconde a quantidade de falsos positivos.")
    if pts:
        doc.add("## Pontos para a discussão", *[f"- {p}" for p in pts])


def build_consolidated(reports: Path = REPORTS_DIR, out: Path | None = None,
                       artifacts: Path = ARTIFACTS_DIR, data_dir: Path = DATA_DIR) -> Path:
    out = out or reports / "consolidado"
    doc = Document(out)
    facts: dict = {}
    doc.add("# Resultados consolidados",
            "Gerado por `python -m fraude consolidate` a partir de `reports/` e `artifacts/`. "
            "As tabelas estão também em `tabelas/` (CSV com vírgula decimal) e as figuras em `figuras/`.")
    section_bases(doc, reports)
    section_duplicates(doc, reports, facts, data_dir)
    for ds in DATASETS:
        section_comparison(doc, reports, ds, facts)
    section_shortcut(doc, reports, facts, data_dir)
    section_statistics(doc, reports, facts)
    section_final(doc, reports, facts, artifacts)
    section_explain(doc, reports, facts)
    section_drift(doc, reports, facts)
    section_discussion(doc, facts)
    path = out / "resultados.md"
    path.write_text("\n".join(doc.lines), encoding="utf-8")
    log.info("%d tabelas e %d figuras em %s", doc.n_tab, doc.n_fig, out)
    return path
