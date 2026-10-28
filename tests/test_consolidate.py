import pandas as pd

from fraude.consolidate import br, build_consolidated, selections


def test_br_formats_brazilian_numbers():
    assert br(1234.5678, 2) == "1.234,57"
    assert br(9144) == "9.144"
    assert br(float("nan")) == "—"


def test_selections_original_vs_revised():
    comp = pd.DataFrame({"config": ["ens@ensemble", "xgb__w@busca50", "mlp__s@experiments"],
                         "auc_pr_mean": [0.857, 0.856, 0.824], "equivalent_to_best": [True, True, True],
                         "latency_mean_ms": [2.9, 2.05, 1.54]})
    sel = selections(comp)
    assert sel == {"original": "mlp__s@experiments", "revisada": "xgb__w@busca50", "melhor": "ens@ensemble"}


def test_build_consolidated_tolerates_missing_results(tmp_path):
    path = build_consolidated(reports=tmp_path, out=tmp_path / "consolidado",
                              artifacts=tmp_path, data_dir=tmp_path)
    assert path.read_text(encoding="utf-8").startswith("# Resultados consolidados")
