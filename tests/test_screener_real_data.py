"""Slow check on the real tables: on every fixed profile, layer row and method,
the screener's scores give the study's tail-loss AUC on its evaluation universe.

Kept out of CI (it needs the processed tables); run with `make test-slow`.
"""

import pandas as pd
import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.evaluation.ablation import LAYERS, ranking_metrics
from ai_mcdm_invest.mcdm.ranking import PROFILES, criterion_weights
from ai_mcdm_invest.screener import METHODS, load_table, score

pytestmark = pytest.mark.slow


def test_the_screener_s_scores_reproduce_the_study_s_auc() -> None:
    processed = Settings.from_env().processed_dir
    ablation_path = processed / "test_year_ablation.parquet"
    if not ablation_path.exists():
        pytest.skip(f"test-year results not found in {processed}")
    ablation = pd.read_parquet(ablation_path)
    ablation = ablation[ablation["sample"] == "all"].set_index(
        ["layer", "profile", "method", "fiscal_year"]
    )
    table, _, criteria, has_text = load_table(processed)
    assert has_text

    checked = 0
    for layer, layers in LAYERS.items():
        for profile in PROFILES:
            weights = criterion_weights(profile, criteria, layers)
            for method in METHODS:
                scored = table.assign(score=score(table, weights, method))
                universe = scored[scored["in_study"]]
                for year, rows in universe.groupby("fiscal_year"):
                    study = ablation.loc[(layer, profile, method, str(year))]
                    metrics = ranking_metrics(rows)
                    assert metrics["companies"] == study["companies"]
                    assert metrics["tail_loss_auc"] == pytest.approx(study["tail_loss_auc"])
                    checked += 1
    assert checked == len(LAYERS) * len(PROFILES) * len(METHODS) * 2
