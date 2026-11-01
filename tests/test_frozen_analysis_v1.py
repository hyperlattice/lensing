import numpy as np

from scripts.score_paired_evaluation_v1 import score_contrast
from src.ml.frozen_analysis_v1 import BOOTSTRAP,contrast_specification


def test_better_primary_mae_cannot_hide_failed_canonical_utility(monkeypatch):
    monkeypatch.setitem(BOOTSTRAP,"resamples",100)
    a=np.full((3,5,9,2),.03)
    b=np.full_like(a,.06)
    a[:,:,0,:]=.12
    result=score_contrast(a,b,contrast_specification()[0])
    assert result["passed_components"]["meaningful_superiority"]
    assert not result["passed_components"]["canonical_A"]
    assert not result["superiority_claim_gate_passed"]
