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
- NB5 xuất GGUF vào `/tmp/lab22-gguf` (không phải `gguf/`) vì `/kaggle/working` chỉ có ~20 GB
  và đã chứa bản `sft-merged` ~8 GB; export cần thêm ~17 GB.
- Trước ô chấm NB4 có thêm một ô **dọn bộ nhớ GPU**: nếu Kaggle cấp 2 GPU nó chuyển reward
  model sang GPU rảnh (tránh `CUDA out of memory`), và tự dựng lại `records` sau khi restart.

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


# --- Kaggle-only fixes applied on top of the Colab render --------------------
# 1. GGUF export: /kaggle/working is a ~20 GB quota that already holds the 8 GB
#    sft-merged, but the export needs ~17 GB more (16-bit merge + f16 GGUF +
#    q4_k_m coexist). Redirect it to the container root FS; the .gguf is
#    gitignored, only data/eval/deploy_meta.json is graded.
GGUF_OLD = '''model.save_pretrained_gguf(str(C.GGUF_DIR), tokenizer, quantization_method="q4_k_m")
# Optional for the +3 rigor add-on: quantization_method=["q4_k_m", "q5_k_m", "q8_0"]


def find_gguf(pattern: str = "q4_k_m") -> Path:
    hits = [p for p in C.REPO_ROOT.glob("gguf*/**/*.gguf") if pattern in p.name.lower()]
    assert hits, f"No *{pattern}*.gguf under {C.REPO_ROOT}/gguf*"
    return max(hits, key=lambda p: p.stat().st_mtime)


gguf_path = find_gguf()
print(f"{gguf_path.relative_to(C.REPO_ROOT)}  {gguf_path.stat().st_size / 1e9:.2f} GB")'''

GGUF_NEW = '''# Kaggle: /kaggle/working is a ~20 GB quota and already holds the 8 GB sft-merged,
# but the export needs ~17 GB more (16-bit merge + f16 GGUF + q4_k_m coexist).
# Write to the container root FS instead; the .gguf is gitignored anyway.
import shutil

GGUF_TMP = Path("/tmp/lab22-gguf")
GGUF_TMP.mkdir(parents=True, exist_ok=True)
print(f"{GGUF_TMP}: {shutil.disk_usage(GGUF_TMP).free / 1e9:.1f} GB free")

model.save_pretrained_gguf(str(GGUF_TMP), tokenizer, quantization_method="q4_k_m")

hits = [p for p in GGUF_TMP.glob("**/*.gguf") if "q4_k_m" in p.name.lower()]
assert hits, f"No *q4_k_m*.gguf under {GGUF_TMP}"
gguf_path = max(hits, key=lambda p: p.stat().st_mtime)
print(f"{gguf_path}  {gguf_path.stat().st_size / 1e9:.2f} GB")'''

DEPLOY_OLD = '    "gguf_path": str(gguf_path.relative_to(C.REPO_ROOT)),'
DEPLOY_NEW = '    "gguf_path": str(gguf_path),  # absolute: the GGUF lives on /tmp, not under the repo'

REPLACEMENTS = [(GGUF_OLD, GGUF_NEW), (DEPLOY_OLD, DEPLOY_NEW)]

# 2. Inserted before the NB4 judge cell: the generation cell can leave ~7 GiB
#    referenced on GPU 0, so each reward model (~8 GiB fp16) no longer fits.
#    Free what we can and, when a second GPU is idle, run the judge there.
JUDGE_MARKER = "provider = C.JUDGE_PROVIDER"

RECOVERY_MD = """### Dọn bộ nhớ GPU trước khi chấm

Ô sinh câu trả lời (NB4 §1) có thể còn giữ ~7 GiB trên GPU 0, khiến reward model
(~8 GiB fp16 trên T4) không nạp nổi — đó là nguyên nhân `CUDA out of memory` ở ô chấm.
Ô dưới đây giải phóng bộ nhớ và, nếu Kaggle cấp 2 GPU, chuyển giám khảo sang GPU đang rảnh.
Nó cũng tự dựng lại `records` từ `side_by_side.jsonl` nếu bạn vừa restart kernel."""

RECOVERY_CODE = '''# Free the previous stage's GPU memory; route the reward-model judge to an idle
# GPU when one exists. Also rebuilds `records` after a kernel restart.
import gc, hashlib, importlib, json
from pathlib import Path
import torch

for _n in ("model", "trainer", "ref_model", "llm", "policy", "ref", "tokenizer", "score"):
    globals().pop(_n, None)
gc.collect()
torch.cuda.empty_cache()

if "records" not in globals():
    _sb = C.EVAL_DIR / "side_by_side.jsonl"
    records = [json.loads(line) for line in _sb.read_text(encoding="utf-8").splitlines() if line.strip()]
    OUTPUTS_SHA = hashlib.sha256(_sb.read_bytes()).hexdigest()
    print(f"rebuilt {len(records)} records from {_sb.name}")

target = 0
if torch.cuda.device_count() > 1 and torch.cuda.mem_get_info(1)[0] > torch.cuda.mem_get_info(0)[0]:
    target = 1
    _jf = Path("/kaggle/working/lab22/lab22/judge.py")
    _jf.write_text(_jf.read_text().replace('device_map="cuda:0"', f'device_map="cuda:{target}"'))
    import lab22.judge as J
    importlib.reload(J)
for _i in range(torch.cuda.device_count()):
    _free, _total = torch.cuda.mem_get_info(_i)
    print(f"  GPU {_i}: {_free / 1e9:.1f} / {_total / 1e9:.1f} GiB free")
print(f"judge will run on cuda:{target}")'''


def set_source(cell: dict, text: str) -> None:
    lines = text.splitlines(keepends=True)
    if lines:
        lines[-1] = lines[-1].rstrip("\n")
    cell["source"] = lines


def render_kaggle() -> dict:
    nb = render("T4")
    for cell in nb["cells"]:
        text = "".join(cell["source"]).replace(COLAB_WORKDIR, KAGGLE_WORKDIR)
        for old, new in REPLACEMENTS:
            text = text.replace(old, new)
        set_source(cell, text)
    nb["cells"][0] = md(INTRO)
    idx = next(i for i, c in enumerate(nb["cells"]) if "".join(c["source"]).lstrip().startswith(JUDGE_MARKER))
    nb["cells"][idx:idx] = [md(RECOVERY_MD), code(RECOVERY_CODE)]
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
