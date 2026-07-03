"""Dataset path configuration.

All local filesystem paths live in a YAML file (see configs/paths.example.yaml)
instead of being hardcoded, so the same code runs against any paired
Visium HD / Visium V2 sample without editing source files.
"""

import os
from dataclasses import dataclass
from typing import List, Optional

import yaml


def _find_first_existing(paths: List[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


@dataclass
class DatasetPaths:
    uni_weights: str

    hd_slide_dir: str
    hd_image: str
    hd_h5_002um: str
    hd_h5_008um: str
    hd_h5_016um: str

    v2_dir: str
    v2_image: str
    v2_positions: str

    output_dir: str

    def hd_coords_candidates(self, bin_name: str) -> List[str]:
        return [
            os.path.join(self.hd_slide_dir, "binned_outputs", bin_name, "spatial", "tissue_positions.parquet"),
            os.path.join(self.hd_slide_dir, "binned_outputs", bin_name, "spatial", "tissue_positions.csv"),
            os.path.join(self.hd_slide_dir, bin_name, "spatial", "tissue_positions.parquet"),
            os.path.join(self.hd_slide_dir, bin_name, "spatial", "tissue_positions.csv"),
            os.path.join(self.hd_slide_dir, bin_name, "spatial", "tissue_positions_list.csv"),
        ]

    def v2_expression_candidates(self) -> List[str]:
        return [
            os.path.join(self.v2_dir, "filtered_feature_bc_matrix.h5"),
            os.path.join(self.v2_dir, "filtered_feature_bc_matrix"),
            os.path.join(self.v2_dir, "raw_feature_bc_matrix.h5"),
            os.path.join(self.v2_dir, "raw_feature_bc_matrix"),
        ]

    def resolve_required_files(self) -> dict:
        """Resolve every path this pipeline needs and raise if any are missing."""
        resolved = {
            "uni_weights": self.uni_weights,
            "hd_image": self.hd_image,
            "hd_h5_002um": self.hd_h5_002um,
            "hd_h5_008um": self.hd_h5_008um,
            "hd_h5_016um": self.hd_h5_016um,
            "hd_coords_002um": _find_first_existing(self.hd_coords_candidates("square_002um")),
            "hd_coords_008um": _find_first_existing(self.hd_coords_candidates("square_008um")),
            "hd_coords_016um": _find_first_existing(self.hd_coords_candidates("square_016um")),
            "v2_image": self.v2_image,
            "v2_positions": self.v2_positions,
            "v2_expression": _find_first_existing(self.v2_expression_candidates()),
        }
        missing = [f"{k}: {v}" for k, v in resolved.items() if not v or not os.path.exists(v)]
        if missing:
            raise FileNotFoundError(
                "Missing required dataset files (edit your paths config):\n" + "\n".join(missing)
            )
        return resolved


def load_dataset_paths(config_path: str) -> DatasetPaths:
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    return DatasetPaths(
        uni_weights=raw["uni_weights"],
        hd_slide_dir=raw["hd"]["slide_dir"],
        hd_image=raw["hd"]["image"],
        hd_h5_002um=raw["hd"]["h5_002um"],
        hd_h5_008um=raw["hd"]["h5_008um"],
        hd_h5_016um=raw["hd"]["h5_016um"],
        v2_dir=raw["v2"]["dir"],
        v2_image=raw["v2"]["image"],
        v2_positions=raw["v2"]["positions"],
        output_dir=raw.get("output_dir", "./outputs"),
    )
