#!/usr/bin/env python3
"""Build a Kaggle-flavoured bundle from the same sources as the Colab notebook.

Reuses `build_colab.render("T4")` (single source of truth), then:

  * rewrites the working directory `/content/lab22` -> `/kaggle/working/lab22`
  * swaps the Colab metadata for Kaggle-friendly metadata
  * appends a final cell that zips every graded artifact for download

    python scripts/build_kaggle.py            # write kaggle/Lab22_DPO_Kaggle.ipynb
    python scripts/build_kaggle.py --check    # exit 1 if it is stale
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_colab import render  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
COLAB_WORKDIR = "/content/lab22"
KAGGLE_WORKDIR = "/kaggle/working/lab22"
TARGET = REPO / "kaggle" / "Lab22_DPO_Kaggle.ipynb"

INTRO = """# Lab 22 — DPO/ORPO Alignment (Kaggle, T4 tier)

**Track 3 · Day 22 · VinUni AICB.** Bản Kaggle của `colab/Lab22_DPO_T4.ipynb`, sinh bằng
`scripts/build_kaggle.py`; đừng sửa tay.

**Khác biệt so với Colab:**

- Thư mục làm việc là `/kaggle/working/lab22` (không phải `/content/lab22`).
- Trong **Session options** phải bật **Internet** (để tải mô hình/dữ liệu từ Hugging Face)
  và chọn **Accelerator = GPU T4 x2** (hoặc T4 x1).
- Kaggle lưu `/kaggle/working` thành output khi **Save Version → Save & Run All**, nhưng
  **vẫn xoá** khi hết phiên. Chạy ô **B. Xuất kết quả** ở cuối ngay khi NB4 xong để lấy
  `lab22-submission.zip`.
- Nếu `pip install` làm hỏng torch/CUDA của Kaggle, chạy lại kernel rồi bỏ các gói đã có sẵn
  (torch, transformers, datasets, accelerate, peft, bitsandbytes) khỏi ô cài đặt.

Core: NB0 → NB4. Bonus: NB3b (variants), NB5 (GGUF), NB6 (lm-eval), NB7 (GRPO).
Each stage reloads what it needs from disk, so after a crash you can restart the runtime,
rerun section A, and continue from the stage that failed.
"""

EXTRACT_MD = """## B. Xuất kết quả (Kaggle)

Kaggle xoá `/kaggle/working` khi hết phiên, giống Colab xoá `/content`. Ô dưới đây gom mọi
artifact được chấm vào **một file zip** ở `/kaggle/working` để tải về (tab **Output** →
`lab22-submission.zip`). Chạy ô này **ngay khi NB4 xong**, đừng đợi phiên kết thúc.
"""

EXTRACT_CODE = '''# ── Gom artifact được chấm vào một file zip ───────────────────────────────────
import json
import zipfile
from pathlib import Path

REPO = Path("/kaggle/working/lab22")
ZIP_PATH = Path("/kaggle/working/lab22-submission.zip")

# Core artifacts (NB1–NB4). Paths are relative to REPO.
CORE = [
    "submission/screenshots/02-sft-loss.png",            # NB1
    "adapters/sft-mini/adapter_config.json",             # NB1
    "models/sft-merged/config.json",                     # NB1 (config only, not weights)
    "submission/screenshots/02b-pref-length.png",        # NB2
    "data/pref/train.parquet",                           # NB2
    "data/pref/eval.parquet",                            # NB2
    "data/pref/stats.json",                              # NB2
    "submission/screenshots/03-dpo-reward-curves.png",   # NB3
    "adapters/dpo/adapter_config.json",                  # NB3
    "adapters/dpo/dpo_metrics.json",                     # NB3
    "adapters/dpo/split.json",                           # NB3
    "submission/screenshots/04-side-by-side-table.png",  # NB4
    "data/eval/side_by_side.jsonl",                      # NB4
    "data/eval/judge_summary.json",                      # NB4
]
# Bonus artifacts, packed when present.
BONUS = [
    "submission/screenshots/03b-variants.png",
    "submission/screenshots/06-gguf-smoke.png",
    "submission/screenshots/07-benchmark-comparison.png",
    "submission/screenshots/08-grpo-reward.png",
    "submission/screenshots/bonus-beta-sweep.png",
    "adapters/variants/variants_summary.json",
    "adapters/grpo/grpo_metrics.json",
    "data/eval/deploy_meta.json",
    "data/eval/benchmark_results.json",
    "data/eval/judge_results_rm.json",
    "data/eval/judge_results_api.json",
]

packed = [p for p in CORE + BONUS if (REPO / p).is_file()]
packed += [str(p.relative_to(REPO)) for p in sorted(REPO.glob("adapters/dpo-b*/dpo_metrics.json"))]
packed = sorted(set(packed))
missing = [p for p in CORE if not (REPO / p).is_file()]

manifest = {"workdir": str(REPO), "packed": packed, "missing_core": missing}
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in packed:
        z.write(REPO / rel, arcname=rel)
    z.writestr("_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))

print(f"Zipped {len(packed)} files -> {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")
if missing:
    print("MISSING core artifacts (rerun the stage that writes them):")
    for p in missing:
        print("  -", p)
else:
    print("OK: every core artifact is present")

# Headline numbers, so REFLECTION can be filled even without the JSONs open.
for rel in ("adapters/dpo/dpo_metrics.json", "data/eval/judge_summary.json"):
    path = REPO / rel
    if path.is_file():
        print(f"\\n-- {rel} --")
        print(path.read_text(encoding="utf-8")[:4000])
'''


def code(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines}


def md(text: str) -> dict:
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def render_kaggle() -> dict:
    nb = render("T4")
    for cell in nb["cells"]:
        cell["source"] = [line.replace(COLAB_WORKDIR, KAGGLE_WORKDIR) for line in cell["source"]]
    nb["cells"][0] = md(INTRO)
    nb["cells"] += [md(EXTRACT_MD), code(EXTRACT_CODE)]
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the Kaggle bundle is stale")
    args = parser.parse_args()
    nb = render_kaggle()
    if args.check:
        if not TARGET.exists() or json.loads(TARGET.read_text(encoding="utf-8")) != nb:
            print(f"stale: {TARGET.name}; run `python scripts/build_kaggle.py`")
            return 1
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {TARGET.relative_to(REPO)} ({len(nb['cells'])} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
