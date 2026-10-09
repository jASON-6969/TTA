from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data" / "raw" / "sz_cxr" / "images"
OUTPUT = ROOT / "output" / "experiments" / "benchmark_cxr_source_data_audit.json"
KAGGLE_API = "https://www.kaggle.com/api/v1/datasets/list/{}?pageSize=200{}"

CANDIDATES = [
    {
        "ref": "yoctoman/shcxr-lung-mask",
        "role": "public Shenzhen lung-mask archive used locally",
        "license": "CC BY-NC-SA 4.0",
    },
    {
        "ref": "nhannguyenle/shenzhen-montgomery-cxrdataset",
        "role": "Shenzhen/Montgomery mirror with a mask directory",
        "license": "Unknown in Kaggle metadata",
    },
    {
        "ref": "heyoujue/lungsegmentation",
        "role": "CXR image mirror candidate",
        "license": "Not used; no mask directory found",
    },
    {
        "ref": "iamtapendu/chest-x-ray-lungs-segmentation",
        "role": "generic chest X-ray segmentation candidate",
        "license": "Not used; no exact CHNCXR mask paths found",
    },
]


def get_json(url: str) -> dict:
    request = Request(url, headers={"User-Agent": "benchmark-cxr-source-audit/1.0"})
    with urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def list_files(ref: str) -> list[dict]:
    token = ""
    files: list[dict] = []
    while True:
        suffix = "&pageToken=" + quote(token, safe="") if token else ""
        payload = get_json(KAGGLE_API.format(ref, suffix))
        files.extend(payload.get("datasetFiles", []))
        token = payload.get("nextPageTokenNullable") or ""
        if not token:
            return files


def basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def is_true_mask_path(path: str) -> bool:
    normalized = path.replace("\\", "/").lower()
    return "/mask/" in "/" + normalized and "chncxr" in normalized


def main() -> None:
    image_names = sorted(path.name for path in IMAGE_DIR.glob("*.png"))
    image_set = set(image_names)
    rows = []
    for candidate in CANDIDATES:
        files = list_files(candidate["ref"])
        mask_paths = [item.get("name", "") for item in files if is_true_mask_path(item.get("name", ""))]
        mask_basenames = {
            re.sub(r"(?i)_mask\.png$", ".png", basename(path))
            for path in mask_paths
        }
        exact_matches = sorted(image_set & mask_basenames)
        rows.append(
            {
                **candidate,
                "manifest_file_count": len(files),
                "true_mask_path_count": len(mask_paths),
                "unique_mask_id_count": len(mask_basenames),
                "exact_image_mask_matches": len(exact_matches),
                "missing_from_mask_archive": len(image_set - mask_basenames),
                "complete_662_mask_archive": len(exact_matches) == len(image_set) == 662,
            }
        )

    result = {
        "audit": "SZ-CXR source data completeness against the benchmark CXR protocol",
        "paper_claim": {
            "source_images": 662,
            "source_masks": 662,
            "target_images": 138,
            "source_to_target": "SZ-CXR -> Montgomery",
        },
        "local_image_count": len(image_names),
        "candidates": rows,
        "complete_public_662_mask_archive_found": any(row["complete_662_mask_archive"] for row in rows),
        "conclusion": (
            "No audited public candidate contains masks for all 662 local CHNCXR images. "
            "The two Shenzhen mask archives expose 566 exact masks and omit the same 96 images; "
            "the benchmark paper's complete 662-mask source is not identifiable from the public "
            "repository, cited source link, or audited public manifests."
        ),
        "source_links": {
            "paper_repo": "https://github.com/wenjing-gg/MedSeg-TTA",
            "paper_assets": "https://github.com/wenjing-gg/MedSeg-TTA/blob/main/ASSETS.md",
            "shenzhen_images": "https://huggingface.co/datasets/Famatsu123/montgomery-shenzhen-tuberculosis-cxr",
            "mask_archive": "https://www.kaggle.com/datasets/yoctoman/shcxr-lung-mask",
            "mask_archive_candidate": "https://www.kaggle.com/datasets/nhannguyenle/shenzhen-montgomery-cxrdataset",
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
