"""Integrated arm controls on explicit unit fixtures, never final images."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.observation_bank_v1 import ObservationBank
from src.ml.robustness_training_v1 import ARMS, train_arm
from src.simulation.adaptive_integration_v1 import adaptive_clean, pixel_fluxes
from src.simulation.precise_point_renderer_v1 import fourth_order_amplitudes


def fixture_bank(root, role, seed):
    root.mkdir()
    records, targets = [], {}
    for i in range(2):
        parent = f"{role}-parent-{i}"
        products = {}
        keys = ["base", "pixel-scale-004", "pixel-scale-006", "gaussian-fwhm-075", "gaussian-fwhm-125"]
        for j, k in enumerate(keys):
            shape = (40,40) if k=="pixel-scale-004" else (27,27) if k=="pixel-scale-006" else (32,32)
            a = np.full(shape, .3+i*.2, dtype=np.float32)
            p = root/f"{i}-{k}.npy"
            np.save(p,a)
            products[k] = {"path": p.name}
        p = {"parent_system_id": parent, "role": role, "parameter_seed": seed+i,
             "noise_seed": seed+100+i, "ancestry_ids": [], "latent_id": f"{role}-{i}"}
        if role=="target_adaptation":
            p["images"] = [products["base"]]
        else:
            p.update(expected=products, detector={"exposure_s":100.,"read_noise_electrons":2.,"background_rate":.03},
                     historical_condition={"pixel_scale_arcsec": .04, "read_noise_multiplier":2., "exposure_multiplier":1.})
            targets[parent] = .8+i*.4
        records.append(p)
    label = root/"targets.json"
    label.write_text(json.dumps(targets))
    m = {"schema":"observation_bank_v1", "stage":"unit_fixture", "role":role, "parents":records,
         "exclusion_ledger":[],"condition_access":"gaussian_only",
         "targets": None if role=="target_adaptation" else {"path":label.name}}
    path = root/"bank.json"
    path.write_text(json.dumps(m))
    return path


def test_qualified_bank_loads_without_historical_renderer_source(tmp_path):
    path = fixture_bank(tmp_path / "target", "target_adaptation", 1000)
    report = tmp_path / "qualification.json"
    report.write_text(json.dumps({"passed": True, "rendering_factor": "fixture"}))
    document = json.loads(path.read_text())
    document.update(stage="qualified_development", rendering_factor="fixture",
                    image_unit="electrons_per_second",
                    target_definition="lenstronomy_SIE_circularized_parameter_arcsec",
                    qualification={"path": str(report)})
    path.write_text(json.dumps(document))
    bank = ObservationBank(path, expected_role="target_adaptation")
    np.testing.assert_array_equal(bank.target_image(bank.parents[0]["parent_system_id"], 0),
                                  np.full((32, 32), .3, dtype=np.float32))


def test_all_eight_arms_share_init_parent_exposure_and_reject_final_roles(tmp_path, monkeypatch):
    from src.ml.model import build_model
    initial_states=[]
    def capture_model(config):
        model=build_model(config)
        initial_states.append({k:v.clone() for k,v in model.state_dict().items()})
        return model
    monkeypatch.setattr("src.ml.robustness_training_v1.build_model",capture_model)
    source = ObservationBank(fixture_bank(tmp_path/"source","source_train",1000),expected_role="source_train",allow_fixture=True)
    dev = ObservationBank(fixture_bank(tmp_path/"dev","source_development",2000),expected_role="source_development",allow_fixture=True)
    target = ObservationBank(fixture_bank(tmp_path/"target","target_adaptation",3000),expected_role="target_adaptation",allow_fixture=True)
    config = {"optimizer_updates":4,"validation_interval_updates":2,"batch_size":2,"torch_threads":2,
              "model":{"channels":[2,4,4,4],"pooled_size":1,"hidden_features":8,"dropout":0.},
              "learning_rate":.001,"weight_decay":.0001,"huber_delta":1.,"mmd_weight":.1}
    reports = []
    for arm in sorted(ARMS):
        reports.append(train_arm(config,seed=720401,arm=arm,source=source,development=dev,
            target=target if "+MMD" in arm else None,output=tmp_path/arm,
            probabilities=[.7,.05,.05,.05,.05,.05,.05],allow_fixture=True))
    schedules=[json.loads((tmp_path/arm/"sampling_controls.json").read_text())["parent_schedule"] for arm in sorted(ARMS)]
    assert all(schedule==schedules[0] for schedule in schedules)
    for state in initial_states[1:]:
        for key in state:
            torch.testing.assert_close(state[key],initial_states[0][key],rtol=0,atol=0)
    assert {r["source_exposures"] for r in reports}=={8}
    assert all(r["target_exposures"]==(8 if "+MMD" in r["arm"] else 0) for r in reports)
    assert all(np.isfinite(r["validation_mae_arcsec"]) and not r["final_images_accessed"] for r in reports)
    with pytest.raises(ValueError,match="Source-only"):
        train_arm(config,seed=1,arm="C",source=source,development=dev,target=target,output=tmp_path/"bad",probabilities=None,allow_fixture=True)
    with pytest.raises(ValueError,match="pilot/final"):
        ObservationBank(source.path,expected_role="final",allow_fixture=True)


def test_fresh_detector_draws_and_source_only_normalization(tmp_path):
    path = fixture_bank(tmp_path/"source","source_train",1000)
    with pytest.raises(ValueError,match="fixtures"):
        ObservationBank(path,expected_role="source_train")
    bank=ObservationBank(path,expected_role="source_train",allow_fixture=True)
    p=next(iter(bank.by_id))
    np.testing.assert_array_equal(bank.source_image(p),bank.source_image(p))
    assert not np.array_equal(bank.source_image(p,noise_seed=8),bank.source_image(p,noise_seed=9))
    dev=ObservationBank(fixture_bank(tmp_path/"dev","source_development",2000),expected_role="source_development",allow_fixture=True)
    with pytest.raises(ValueError,match="source-training-only"):
        dev.source_statistics()


def test_latent_alias_cannot_cross_source_roles_with_new_ids_and_seeds(tmp_path):
    source=ObservationBank(fixture_bank(tmp_path/"source","source_train",1000),expected_role="source_train",allow_fixture=True)
    path=fixture_bank(tmp_path/"dev","source_development",2000)
    document=json.loads(path.read_text())
    document["parents"][0]["latent_id"]=source.parents[0]["latent_id"]
    path.write_text(json.dumps(document))
    dev=ObservationBank(path,expected_role="source_development",allow_fixture=True)
    cfg={"optimizer_updates":1,"validation_interval_updates":1,"batch_size":2,"torch_threads":2,"mmd_weight":0.}
    with pytest.raises(ValueError,match="fingerprint overlaps"):
        train_arm(cfg,seed=1,arm="C",source=source,development=dev,target=None,output=tmp_path/"rejected-fit",probabilities=None,allow_fixture=True)


class PolynomialEngine:
    def alpha(self, xy):
        return np.zeros_like(xy)
    def light(self, prefix, xy):
        return np.zeros(len(xy)) if prefix=="lens_light" else 1+xy[:,0]**2+xy[:,1]**2
    def clean(self, c, *, factor):
        h,w=c["shape"]
        return pixel_fluxes(self,c,np.arange(h*w),factor=factor).reshape(h,w)


def test_adaptive_pixels_match_an_independent_polynomial_integral():
    e=PolynomialEngine()
    c={"shape":[10,12],"pixel_scale_arcsec":.1}
    result,diag=adaptive_clean(e,c,relative_tolerance=1e-4)
    yy,xx=np.indices((10,12))
    x,y=(xx-5.5)*.1,(yy-4.5)*.1
    truth=.1**2*(1+x*x+y*y+.1**2/6)
    assert abs(result-truth).sum()/truth.sum()<1e-4
    assert diag["passed_internal_estimate"]
    with pytest.raises(ValueError,match="Unresolved"):
        adaptive_clean(e,c,relative_tolerance=1e-12,maximum_factor=8)


def test_point_jacobian_matches_closed_form_sis_magnification():
    class SIS:
        p={"point_source_amp":.8}
        def alpha(self,xy):
            return xy/np.linalg.norm(xy,axis=1)[:,None]
    positions=np.array([[1.3,0.],[0.,.7],[.8,.8]])
    truth=.8/abs(1-1/np.linalg.norm(positions,axis=1))
    values=fourth_order_amplitudes(SIS(),positions)
    np.testing.assert_allclose(values,truth,rtol=1e-8,atol=1e-10)


def test_isolated_simulation_condition_definitions_match_preserved_ml_policy():
    from src.ml.augmentation_policy_v1 import CELLS as ml_cells,condition_for as ml_condition
    from src.processing.observation_conditions_v1 import CELLS as data_cells,condition_for as data_condition
    assert ml_cells==data_cells
    for cell in ml_cells:
        assert ml_condition(cell)==data_condition(cell)
