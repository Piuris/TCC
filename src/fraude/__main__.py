"""Interface de linha de comando: ``python -m fraude <comando>``.

Exemplos::

    python -m fraude download --dataset creditcard
    python -m fraude eda --dataset creditcard
    python -m fraude experiment --dataset creditcard --n-iter 10
    python -m fraude report --dataset creditcard
    python -m fraude train-final --dataset creditcard
    python -m fraude explain --dataset creditcard
    python -m fraude drift --dataset creditcard
    python -m fraude all --dataset synthetic --quick
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

from .config import DATA_DIR, REPORTS_DIR, ExperimentConfig, ensure_dirs, set_global_seed
from .data import PAYSIM_FILE, SYNTHETIC_FILE, ULB_FILE, load_dataset, make_synthetic_creditcard
from .models import BALANCING, MODELS

log = logging.getLogger("fraude")


ULB_OPENML_FILE_ID = 1673544


def _fetch_ulb_arff():
    """Baixa a base ULB do OpenML lendo o ARFF bruto.

    ``fetch_openml`` descarta a coluna ``Time`` por ela estar marcada como
    ``row_id_attribute`` no OpenML; o ARFF bruto a mantém. Reaproveita o cache
    do scikit-learn quando existe.
    """
    import gzip
    import io
    import urllib.request

    import pandas as pd

    cache = Path.home() / "scikit_learn_data/openml/openml.org/data/v1/download" / f"{ULB_OPENML_FILE_ID}.gz"
    if cache.exists():
        log.info("Usando ARFF em cache: %s", cache)
        raw = gzip.decompress(cache.read_bytes())
    else:
        url = f"https://api.openml.org/data/v1/download/{ULB_OPENML_FILE_ID}"
        log.info("Baixando a base ULB do OpenML (%s, ~150 MB)...", url)
        raw = urllib.request.urlopen(url).read()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
    text = raw.decode("utf-8")
    header, data = text.split("@data", 1) if "@data" in text else text.split("@DATA", 1)
    names = [line.split()[1].strip("'\"") for line in header.splitlines()
             if line.lower().startswith("@attribute")]
    frame = pd.read_csv(io.StringIO(data), names=names, comment="%", skip_blank_lines=True,
                        quotechar="'")
    frame["Class"] = frame["Class"].astype(str).str.strip("'\"").astype(int)
    return frame

def download(dataset: str) -> None:
    ensure_dirs()
    if dataset == "creditcard":
        target = DATA_DIR / ULB_FILE
        if target.exists():
            log.info("Já existe: %s", target)
            return
        frame = _fetch_ulb_arff()
        frame.to_csv(target, index=False)
        log.info("Salvo em %s (%d linhas, %d fraudes)", target, len(frame), int(frame["Class"].sum()))
    elif dataset == "paysim":
        target = DATA_DIR / PAYSIM_FILE
        if target.exists():
            log.info("Já existe: %s", target)
            return
        if not shutil.which("kaggle"):
            sys.exit("O PaySim é distribuído pelo Kaggle. Instale e configure a CLI (kaggle.json) "
                     "ou baixe manualmente https://www.kaggle.com/datasets/ealaxi/paysim1 e salve "
                     f"o CSV como {target}.")
        tmp = DATA_DIR / "_paysim"
        subprocess.run(["kaggle", "datasets", "download", "-d", "ealaxi/paysim1", "-p", str(tmp),
                        "--unzip"], check=True)
        csv = next(tmp.glob("*.csv"))
        csv.replace(target)
        shutil.rmtree(tmp, ignore_errors=True)
        log.info("Salvo em %s", target)
    elif dataset == "synthetic":
        target = DATA_DIR / SYNTHETIC_FILE
        make_synthetic_creditcard().to_csv(target, index=False)
        log.info("Base sintética salva em %s", target)
    else:
        sys.exit(f"Base desconhecida: {dataset}")


def _cfg(args) -> ExperimentConfig:
    cfg = ExperimentConfig()
    if getattr(args, "quick", False):
        cfg.n_iter_search, cfg.cv_folds = 3, 3
    for attr, arg in [("n_iter_search", "n_iter"), ("cv_folds", "cv"), ("seed", "seed"),
                      ("sample_frac", "sample_frac"), ("models", "models"), ("balancing", "balancing")]:
        value = getattr(args, arg, None)
        if value is not None:
            setattr(cfg, attr, value)
    return cfg


def _report_dir(args) -> Path:
    return REPORTS_DIR / args.dataset


def _all_records(args) -> list[dict]:
    """Resultados de todos os diretórios de experimentos (principal, busca ampliada, ensembles)."""
    records = []
    for d in sorted(_report_dir(args).glob("experiments*")):
        f = d / "results.json"
        if f.exists():
            for r in json.loads(f.read_text(encoding="utf-8")):
                records.append({**r, "source": d.name})
    return records


def _candidate_key(r: dict) -> str:
    suffix = "" if r["source"] == "experiments" else "@" + r["source"].removeprefix("experiments_")
    return f"{r['model']}__{r['balancing']}{suffix}"


def _spec(r: dict) -> dict:
    return {"model": r["model"], "balancing": r["balancing"],
            "params": r.get("best_params") or {}, "members": r.get("members")}


def _chosen_config(args) -> dict:
    """Configuração escolhida, em ordem de prioridade:

    1. ``--model`` e ``--balancing-one`` explícitos (hiperparâmetros padrão);
    2. a seleção da comparação estatística (``comparacao/selection.json``);
    3. a maior AUC-PR média na CV entre todos os experimentos.
    """
    if args.model and args.balancing_one:
        return {"model": args.model, "balancing": args.balancing_one, "params": {}, "members": None}
    selection = _report_dir(args) / "comparacao" / "selection.json"
    if selection.exists() and not args.model and not args.balancing_one:
        spec = json.loads(selection.read_text(encoding="utf-8"))["spec"]
        log.info("Configuração escolhida pela comparação estatística: %s + %s",
                 spec["model"], spec["balancing"])
        return spec
    records = _all_records(args)
    if not records:
        log.warning("Sem resultados de experimentos; usando LightGBM + pesos de classe.")
        return {"model": args.model or "lightgbm", "balancing": args.balancing_one or "class_weight",
                "params": {}, "members": None}
    from .experiments import select_best

    if args.model:
        records = [r for r in records if r["model"] == args.model]
    if args.balancing_one:
        records = [r for r in records if r["balancing"] == args.balancing_one]
    if not records:
        sys.exit("Nenhum resultado corresponde ao modelo/balanceamento pedidos.")
    best = select_best(records)
    log.info("Configuração escolhida pela CV: %s + %s (AUC-PR CV=%.4f, %s)", best["model"],
             best["balancing"], best["cv_auc_pr_mean"], best["source"])
    return _spec(best)


def cmd_eda(args):
    from .eda import run_eda

    ds = load_dataset(args.dataset, args.sample_frac, dedupe=False)
    stats = run_eda(ds, _report_dir(args) / "eda")
    print(json.dumps({k: stats[k] for k in ("n_rows", "n_fraud", "fraud_rate_pct",
                                            "trivial_accuracy_pct", "duplicates")}, indent=2))


def cmd_experiment(args):
    from .experiments import run_experiments

    cfg = _cfg(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    log.info("Base: %s", ds.summary())
    table = run_experiments(ds, cfg, _report_dir(args) / args.out_name, resume=not args.no_resume)
    cols = ["model", "balancing", "auc_pr", "auc_roc", "precision", "recall", "f1", "mcc",
            "latency_mean_ms"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cmd_report(args)


def cmd_report(args):
    from .reporting import build_report

    out = _report_dir(args) / getattr(args, "out_name", "experiments")
    build_report(out)
    log.info("Relatório em %s", out / "summary.md")


def cmd_ensemble(args):
    from .ensemble import ensemble_jobs
    from .experiments import run_experiments, split_train_test
    from .reporting import build_report

    cfg = _cfg(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    records = [r for r in _all_records(args) if r["source"] != "experiments_ensemble"]
    tr, _ = split_train_test(ds, cfg)
    y_tr = ds.y.iloc[tr]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    jobs = ensemble_jobs(records, ds.kind, pos_weight, cfg.seed)
    for name, _, kw in jobs:
        log.info("%s: %s", name,
                 [(m["model"], m["balancing"], m["source"]) for m in kw["extra"]["members"]])
    out = _report_dir(args) / "experiments_ensemble"
    table = run_experiments(ds, cfg, out, resume=not args.no_resume, jobs=jobs)
    print(table[["model", "auc_pr", "recall", "precision", "latency_mean_ms"]].to_string(index=False))
    build_report(out)


def cmd_compare(args):
    from . import stats as S
    from .ensemble import build_from_spec
    from .experiments import split_train_test
    from .models import BALANCING_LABELS, MODEL_LABELS

    cfg = _cfg(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    records = _all_records(args)
    if not records:
        sys.exit("Sem resultados de experimentos para comparar.")
    by_key = {_candidate_key(r): r for r in records}
    candidates = {k: _spec(r) for k, r in by_key.items()}

    tr, _ = split_train_test(ds, cfg)
    X_tr, y_tr = ds.X.iloc[tr], ds.y.iloc[tr]
    pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    out = _report_dir(args) / "comparacao"
    out.mkdir(parents=True, exist_ok=True)

    scores = S.repeated_cv_scores(
        lambda spec: build_from_spec(spec, ds.kind, pos_weight, cfg.seed), candidates, X_tr, y_tr,
        n_splits=cfg.cv_folds, n_repeats=args.repeats, seed=cfg.seed,
        out_csv=out / "repeated_cv_scores.csv")
    friedman = S.friedman_nemenyi(scores)
    table = S.compare_to_best(scores, n_splits=cfg.cv_folds)
    best = table.attrs["best"]
    latency = {k: r["latency_mean_ms"] for k, r in by_key.items()}
    chosen = S.select_parsimonious(table, latency)

    table["latency_mean_ms"] = [latency[k] for k in table.index]
    table["test_auc_pr"] = [by_key[k]["test_metrics"]["default"]["auc_pr"] for k in table.index]
    table["test_recall_moderado"] = [by_key[k]["test_metrics"]["cost_moderado"]["recall"]
                                     for k in table.index]
    table["test_precision_moderado"] = [by_key[k]["test_metrics"]["cost_moderado"]["precision"]
                                        for k in table.index]
    table.index.name = "config"
    table.to_csv(out / "comparison.csv")

    labels = {k: f"{MODEL_LABELS[r['model']]} — {BALANCING_LABELS[r['balancing']]}"
                 + ("" if r["source"] == "experiments"
                    else f" [{r['source'].removeprefix('experiments_')}]")
              for k, r in by_key.items()}
    S.plot_ranks(friedman, table, labels, out / "ranking_cd.png")
    selection = {
        "key": chosen,
        "spec": candidates[chosen],
        "best_by_mean": best,
        "rule": "entre as configurações não significativamente piores que a melhor (t corrigido "
                f"de Nadeau-Bengio, Holm, α=0,05) e a no máximo {S.PRACTICAL_MARGIN} de AUC-PR "
                f"dela, a de maior AUC-PR entre as mais rápidas (tolerância {S.LATENCY_TOL_MS} ms)",
        "chosen_auc_pr_mean": float(table.loc[chosen, "auc_pr_mean"]),
        "best_auc_pr_mean": float(table.loc[best, "auc_pr_mean"]),
        "p_holm_vs_best": float(table.loc[chosen, "p_holm"]),
    }
    S.save_json(selection, out / "selection.json")
    S.save_json({"friedman_nemenyi": friedman, "n_folds": int(len(scores)),
                 "repeats": args.repeats, "cv_folds": cfg.cv_folds}, out / "statistical_tests.json")

    md = [
        "# Comparação estatística", "",
        f"CV repetida {cfg.cv_folds}×{args.repeats} no treino ({len(scores)} partições, "
        "as mesmas para todas as configurações).", "",
        f"- Friedman: χ² = {friedman['statistic']:.2f}, p = {friedman['p_value']:.2e}; "
        f"Nemenyi CD = {friedman['critical_difference']:.2f}",
        f"- Melhor média: **{labels[best]}** ({table.loc[best, 'auc_pr_mean']:.4f})",
        f"- Regra: {selection['rule']}",
        f"- Escolhida (parcimônia): **{labels[chosen]}** ({table.loc[chosen, 'auc_pr_mean']:.4f}, "
        f"p Holm vs. melhor = {table.loc[chosen, 'p_holm']:.3f}, "
        f"latência {latency[chosen]:.2f} ms)", "",
        "| configuração | AUC-PR (CV rep.) | ranking médio | p Holm vs. melhor | equivalente | "
        "latência ms | AUC-PR teste |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, row in table.iterrows():
        md.append(f"| {labels[k]} | {row['auc_pr_mean']:.4f} ± {row['auc_pr_std']:.4f} | "
                  f"{row['avg_rank']:.2f} | {row['p_holm']:.3f} | "
                  f"{'sim' if row['equivalent_to_best'] else 'não'} | "
                  f"{row['latency_mean_ms']:.2f} | {row['test_auc_pr']:.4f} |")
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md[:7]))


def cmd_train_final(args):
    from .artifact import train_final

    cfg = _cfg(args)
    spec = _chosen_config(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    path = train_final(ds, spec["model"], spec["balancing"], spec["params"], scenario=args.scenario,
                       cfg=cfg, members=spec.get("members"), calibration=args.calibration,
                       report_dir=_report_dir(args) / "calibracao")
    print(path.with_suffix(".json").read_text(encoding="utf-8"))


def cmd_explain(args):
    from .artifact import Predictor, artifact_path
    from .experiments import split_train_test
    from .explain import run_explainability

    cfg = _cfg(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    path = artifact_path(ds.kind)
    if not path.exists():
        sys.exit(f"Artefato {path} não encontrado. Execute `train-final` antes.")
    pred = Predictor.load(path)
    if pred.bundle["dataset"] != ds.name:
        log.warning("Artefato treinado em '%s', analisando '%s'.", pred.bundle["dataset"], ds.name)
    tr, te = split_train_test(ds, cfg)
    summary = run_explainability(pred.pipeline, ds.X.iloc[tr], ds.X.iloc[te], ds.y.iloc[te],
                                 pred.threshold, _report_dir(args) / "explainability", seed=cfg.seed,
                                 calibrator=pred.bundle.get("calibrator"))
    print(json.dumps(summary["global_top10"], indent=2, ensure_ascii=False))


def cmd_drift(args):
    from .drift import run_drift

    cfg = _cfg(args)
    spec = _chosen_config(args)
    ds = load_dataset(args.dataset, cfg.sample_frac, cfg.seed)
    out = _report_dir(args) / ("drift" if args.label_delay == 0 else f"drift_atraso{args.label_delay}")
    df = run_drift(ds, spec["model"], spec["balancing"], spec["params"], out,
                   train_frac=args.train_frac, n_windows=args.windows,
                   label_delay=args.label_delay, seed=cfg.seed, members=spec.get("members"))
    print(df.pivot(index="window", columns="strategy", values="auc_pr").round(4).to_string())


def cmd_consolidate(args):
    from .consolidate import build_consolidated

    path = build_consolidated()
    log.info("Resultados consolidados em %s", path)


def cmd_all(args):
    download(args.dataset)
    for fn in (cmd_eda, cmd_experiment, cmd_ensemble, cmd_compare, cmd_train_final, cmd_explain,
               cmd_drift):
        log.info("===== %s =====", fn.__name__.removeprefix("cmd_"))
        fn(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m fraude", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, experiment=False, choose=False):
        sp.add_argument("--dataset", default="creditcard", choices=["creditcard", "paysim", "paysim_sem_saldo", "synthetic"])
        sp.add_argument("--sample-frac", type=float, default=None,
                        help="Amostra estratificada da base (ex.: 0.2) para execuções rápidas.")
        sp.add_argument("--seed", type=int, default=None)
        sp.add_argument("--cv", type=int, default=None, help="Nº de partições da CV (padrão 5).")
        sp.add_argument("--quick", action="store_true", help="n_iter=3 e CV com 3 partições.")
        if experiment:
            sp.add_argument("--models", nargs="+", choices=MODELS, default=None)
            sp.add_argument("--balancing", nargs="+", choices=BALANCING, default=None)
            sp.add_argument("--n-iter", type=int, default=None, help="Iterações da busca aleatória.")
            sp.add_argument("--no-resume", action="store_true")
            sp.add_argument("--out-name", default="experiments",
                            help="Subdiretório de resultados (ex.: experiments_busca50).")
        if choose:
            sp.add_argument("--model", choices=MODELS, default=None)
            sp.add_argument("--balancing-one", choices=BALANCING, default=None,
                            help="Balanceamento da configuração a usar.")
            sp.add_argument("--scenario", default="moderado",
                            choices=["conservador", "moderado", "agressivo", "valor_transacao"])
            sp.add_argument("--calibration", default="auto",
                            choices=["auto", "none", "sigmoid", "isotonic"])
            sp.add_argument("--repeats", type=int, default=3,
                            help="Repetições da validação cruzada no comando compare.")
            sp.add_argument("--windows", type=int, default=6)
            sp.add_argument("--train-frac", type=float, default=0.4)
            sp.add_argument("--label-delay", type=int, default=0)

    sp = sub.add_parser("download", help="Baixa uma base de dados.")
    sp.add_argument("--dataset", default="creditcard", choices=["creditcard", "paysim", "paysim_sem_saldo", "synthetic"])
    sp.set_defaults(func=lambda a: download(a.dataset))

    for name, fn, exp, choose, help_ in [
        ("eda", cmd_eda, False, False, "Análise exploratória."),
        ("experiment", cmd_experiment, True, False, "Comparação de algoritmos e balanceamentos."),
        ("report", cmd_report, True, False, "Regenera figuras e resumo dos experimentos."),
        ("ensemble", cmd_ensemble, True, False, "Avalia ensembles das melhores configurações."),
        ("compare", cmd_compare, False, True, "CV repetida, testes estatísticos e seleção final."),
        ("train-final", cmd_train_final, False, True, "Treina e serializa o modelo da API."),
        ("explain", cmd_explain, False, True, "Análise SHAP e LIME do modelo final."),
        ("drift", cmd_drift, False, True, "Análise exploratória de concept drift."),
        ("consolidate", cmd_consolidate, False, False,
         "Tabelas e figuras numeradas de todas as bases (reports/consolidado)."),
        ("all", cmd_all, True, True, "Executa todo o fluxo."),
    ]:
        sp = sub.add_parser(name, help=help_)
        common(sp, experiment=exp, choose=choose)
        sp.set_defaults(func=fn)
    return p


def main(argv=None):
    import warnings

    warnings.filterwarnings("ignore", message="Unknown solver options: iprint")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    ensure_dirs()
    set_global_seed(getattr(args, "seed", None) or 42)
    args.func(args)


if __name__ == "__main__":
    main()
