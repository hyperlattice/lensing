"""Image-only native-grid access to a frozen paired pilot/final release.

No method opens the proposal or target vault. PSFs, masks and variance are
validated provenance products, not implicit predictor features. Study IDs and
explicit observation axes identify the requested release.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.data.benchmark_contract_v1 import contained_path, validate_lineage
from src.processing.evaluation_conditions_v1 import CONDITIONS,RENDERERS,evaluation_condition


class PairedEvaluationImages:
    def __init__(self,manifest_path,*,role,study_id=None):
        self.path=Path(manifest_path)
        self.root=self.path.parent
        self.document=json.loads(self.path.read_text())
        doc=self.document
        fields={"schema","role","qualification",
            "rendering_factor","parent_ids","condition_order","renderer_order","observations","image_unit","variance_unit","mask_semantics","proposal_count","accepted_proposals",
            "physical_single_image_rejections","numerical_failures","noise_coupling"}
        if not fields.issubset(doc) or doc["schema"]!="paired_evaluation_release_v1" or doc["role"]!=role or role not in {"power_pilot","final"}:
            raise ValueError("Unexpected evaluation release schema or role")
        q=doc["qualification"]
        report=json.loads(Path(q["path"]).read_text())
        if report["passed"] is not True or report["rendering_factor"]!=doc["rendering_factor"]:
            raise ValueError("Release is not numerically qualified")
        if (doc["condition_order"]!=list(CONDITIONS) or doc["renderer_order"]!=list(RENDERERS) or
            doc["image_unit"]!="electrons_per_second" or doc["variance_unit"]!="electrons_per_second_squared" or
            doc["mask_semantics"]!="true_is_excluded" or doc["numerical_failures"]!=0):
            raise ValueError("Observation matrix, units or qualification failure differs")
        self.parent_ids=doc["parent_ids"]
        if len(self.parent_ids)<2 or len(set(self.parent_ids))!=len(self.parent_ids):
            raise ValueError("Invalid parent identities")
        row_fields={"parent_system_id","role","parameter_seed","noise_seed","ancestry_ids","latent_id",
            "renderer","condition","shape","pixel_scale_arcsec","psf_family","condition_parameters","detector","products"}
        self.rows={}
        fingerprints={}
        lineage=[]
        for source_row in doc["observations"]:
            row = {**source_row, "latent_id": source_row.get("latent_id", source_row.get("latent_sha256", source_row["parent_system_id"]))}
            if set(row).difference({"latent_sha256"})!=row_fields or row["role"]!=role:
                raise ValueError("Unexpected physical metadata or truth in observation row")
            key=(row["parent_system_id"],row["renderer"],row["condition"])
            if key in self.rows or key[0] not in self.parent_ids or key[1] not in RENDERERS or key[2] not in CONDITIONS:
                raise ValueError("Duplicate or unknown observation identity")
            condition=evaluation_condition(key[2])
            if (row["condition_parameters"]!=condition or row["shape"]!=condition["shape"] or
                row["pixel_scale_arcsec"]!=condition["pixel_scale_arcsec"] or row["psf_family"]!=condition["psf_family"]):
                raise ValueError("Native geometry or condition metadata differs")
            if set(row["products"])!={"image","variance","psf","invalid_mask","coverage_mask"}:
                raise ValueError("Missing required observation product")
            detector=row["detector"]
            if (set(detector)!={"exposure_s","read_noise_electrons","background_rate","noise_seed"} or
                not np.isfinite(list(detector.values())).all() or detector["exposure_s"]<=0 or
                min(detector["read_noise_electrons"],detector["background_rate"],detector["noise_seed"])<0):
                raise ValueError("Invalid explicit detector provenance")
            fingerprint=row["latent_id"]
            if fingerprint in fingerprints and fingerprints[fingerprint]!=key[0]:
                raise ValueError("Physical parent has multiple identities")
            fingerprints[fingerprint]=key[0]
            lineage.append(row)
            self.rows[key]=row
        expected=len(self.parent_ids)*len(CONDITIONS)*len(RENDERERS)
        if len(self.rows)!=expected:
            raise ValueError("Missing paired observations")
        validate_lineage(lineage)
        self._validated=set()

    def _product(self,row,key):
        record=row["products"][key]
        if "path" not in record:
            raise ValueError("Invalid product identity")
        path=contained_path(self.root,record["path"])
        if record["path"] not in self._validated:
            a=np.load(path,allow_pickle=False,mmap_mode="r")
            shape=tuple(row["shape"])
            if key in {"invalid_mask","coverage_mask"}:
                if a.dtype!=np.bool_ or a.shape!=shape or a.any():
                    raise ValueError("Expected all-valid native mask")
            elif a.dtype!=np.float32 or a.ndim!=2 or not np.isfinite(a).all():
                raise ValueError("Invalid finite float32 product")
            elif key=="psf":
                if np.any(a<0) or not np.isclose(a.sum(dtype=float),1.,atol=1e-6):
                    raise ValueError("Invalid normalized PSF")
            elif a.shape!=shape or (key=="variance" and np.any(a<0)):
                raise ValueError("Invalid native image or variance")
            self._validated.add(record["path"])
        return np.load(path,allow_pickle=False,mmap_mode="r")

    def verify_all_products(self):
        for row in self.rows.values():
            for key in row["products"]:
                self._product(row,key)

    def image(self,parent,renderer,condition):
        return self._product(self.rows[(parent,renderer,condition)],"image")
