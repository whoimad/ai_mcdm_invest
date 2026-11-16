"""Slow acceptance check on the real raw data: the rewritten ML model reproduces
the prototype's out-of-time tail-loss AUC (decision 4.7).

Kept out of CI (it needs the raw data); run with `make test-slow`.
"""

import pytest

from ai_mcdm_invest.config import Settings
from ai_mcdm_invest.data.panel import clean_panel, load_raw_panel
from ai_mcdm_invest.data.raw_manifest import load_manifest, verify_raw_files
from ai_mcdm_invest.ml.risk_score import ml_auc_by_year, ml_risk_scores

pytestmark = pytest.mark.slow


def test_the_ml_risk_score_reproduces_the_prototype_s_out_of_time_auc() -> None:
    settings = Settings.from_env()
    if not settings.raw_panel_dir.is_dir():
        pytest.skip(f"real raw data not found in {settings.raw_dir}")
    verify_raw_files(settings.raw_dir, load_manifest(settings.manifest_path))
    clean = clean_panel(load_raw_panel(settings.raw_panel_dir))

    auc = ml_auc_by_year(ml_risk_scores(clean), clean)["tail_loss_auc"]

    # The prototype scored 0.813 (2017) and 0.796 (2018) with sector dummies
    # and last year's move added; on the 221 fundamentals alone the rewrite
    # scores 0.810 and 0.792.
    assert auc[2017] == pytest.approx(0.81, abs=0.01)
    assert auc[2018] == pytest.approx(0.80, abs=0.01)
