"""Native product and image-only contract tests without a truth sidecar."""
import json

import numpy as np
import pytest

from src.data.benchmark_contract_v1 import sha256
from src.data.paired_evaluation_images_v1 import PairedEvaluationImages
from src.processing.evaluation_conditions_v1 import CONDITIONS,RENDERERS,evaluation_condition


def fixture(root):
    def save(name,a):
        path=root/name
        np.save(path,a)
        return {"path":name,"sha256":sha256(path)}
    products={}
    psf=save("psf.npy",np.ones((1,1),np.float32))
    for shape in ((256,256),(320,320),(213,213)):
        products[shape]={"image":save(f"image-{shape[0]}.npy",np.full(shape,.1,np.float32)),
            "variance":save(f"variance-{shape[0]}.npy",np.full(shape,.01,np.float32)),"psf":psf,
            "invalid_mask":save(f"mask-{shape[0]}.npy",np.zeros(shape,bool)),
            "coverage_mask":save(f"coverage-{shape[0]}.npy",np.zeros(shape,bool))}
    qualification=root/"qualification.json"
    qualification.write_text(json.dumps({"passed":True,"rendering_factor":"unit-only","source_sha256":{}}))
    rows=[]
    for parent in range(2):
        for renderer in RENDERERS:
            for name in CONDITIONS:
                c=evaluation_condition(name)
                rows.append({"parent_system_id":f"p{parent}","role":"final","parameter_seed":parent+10,
                    "noise_seed":parent+20,"ancestry_ids":[],"latent_sha256":f"unit-{parent}",
                    "renderer":renderer,"condition":name,"shape":c["shape"],"pixel_scale_arcsec":c["pixel_scale_arcsec"],
                    "psf_family":c["psf_family"],"condition_parameters":c,
                    "detector":{"exposure_s":100.,"read_noise_electrons":2.,"background_rate":.03,"noise_seed":100+parent},
                    "products":products[tuple(c["shape"])]})
    manifest={"schema":"paired_evaluation_release_v1","role":"final","method_freeze_sha256":"fixture-method",
        "evaluation_freeze_sha256":"fixture-evaluation","qualification":{"path":str(qualification),"sha256":sha256(qualification)},
        "rendering_factor":"unit-only","parent_ids":["p0","p1"],"condition_order":list(CONDITIONS),
        "renderer_order":list(RENDERERS),"observations":rows,"truth_sha256":"NO_TRUTH_FILE_EXISTS",
        "image_unit":"electrons_per_second","variance_unit":"electrons_per_second_squared",
        "mask_semantics":"true_is_excluded","proposal_count":2,"accepted_proposals":2,
        "physical_single_image_rejections":0,"numerical_failures":0,"noise_coupling":"fixture",
        "generator_sha256":{},"proposal_audit_sha256":"fixture"}
    path=root/"manifest.json"
    path.write_text(json.dumps(manifest))
    return path,manifest


def test_image_only_loader_never_needs_a_truth_file(tmp_path):
    path,_=fixture(tmp_path)
    images=PairedEvaluationImages(path,role="final",freeze_sha256="fixture-evaluation")
    images.verify_all_products()
    assert images.image("p0","autolens","pixel-scale-004").shape==(320,320)
    assert not (tmp_path/"truth_vault").exists()


def test_truth_injection_and_missing_view_are_rejected(tmp_path):
    path,manifest=fixture(tmp_path)
    manifest["observations"][0]["theta_E_arcsec"]=1.
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="truth"):
        PairedEvaluationImages(path,role="final",freeze_sha256="fixture-evaluation")
    del manifest["observations"][0]["theta_E_arcsec"]
    manifest["observations"].pop()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match="Missing paired"):
        PairedEvaluationImages(path,role="final",freeze_sha256="fixture-evaluation")


def test_changed_native_image_is_rejected(tmp_path):
    path,_=fixture(tmp_path)
    images=PairedEvaluationImages(path,role="final",freeze_sha256="fixture-evaluation")
    np.save(tmp_path/"image-256.npy",np.zeros((256,256),np.float32))
    with pytest.raises(ValueError,match="product changed"):
        images.image("p0","autolens","canonical")
