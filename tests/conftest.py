import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

WORDS = ["xin", "chào", "các", "bạn", "hôm", "nay", "trời", "đẹp", "tôi", "học", "tiếng", "việt"]

TINY_ENCODER = dict(
    output_size=32, attention_heads=4, linear_units=64, num_blocks=3, dropout_rate=0.1,
    encoder_layerdrop=0.0, positional_dropout_rate=0.1, attention_dropout_rate=0.1,
    input_layer="dw_striding", normalize_before=True, cnn_module_kernel=7, use_cnn_module=True,
    activation_type="swish", pos_enc_layer_type="chunk_rel_pos", selfattention_layer_type="chunk_rel_seflattn",
    cnn_module_norm="layer_norm", dynamic_conv=True,
    dynamic_chunk_sizes=[-1, 16, 32], dynamic_left_context_sizes=[16, 32], dynamic_right_context_sizes=[16, 32],
)


def tiny_cfg(**train_overrides):
    """A complete config (same shape as configs/base.yaml) with a tiny random encoder, for CPU tests."""
    from teda.utils.config import AttrDict, deep_merge, load_yaml

    cfg = load_yaml(ROOT / "configs" / "base.yaml")
    cfg = deep_merge(cfg, {
        "encoder": {"encoder_conf": TINY_ENCODER, "input_dim": 80, "overrides": {}, "freeze_embed": True, "freeze_layers": [0]},
        "train": {"epochs": 2, "max_frames_per_batch": 1200, "accum_grad": 1, "num_workers": 0, "warmup_steps": 2,
                  "log_every": 1, "save_every_steps": 1000, "max_hours": None, "lr_encoder": 1e-3, "lr_head": 1e-2,
                  **train_overrides},
        "eval": {"batch_frames": 3000},
        "features": {"spec_aug": {"enabled": True}},
    })
    return AttrDict(cfg)


@pytest.fixture()
def tiny_dataset(tmp_path):
    """Synthetic ViMD-like folder: wav16k/*.wav + manifest.tsv + a tiny BPE tokenizer."""
    import numpy as np
    import soundfile as sf

    from teda.data.tokenizer import train_bpe
    from teda.utils.manifest import FULL_FIELDS, write_tsv

    rng = random.Random(0)
    nrng = np.random.default_rng(0)
    root = tmp_path / "vimd"
    rows = []
    counts = {"train": 24, "valid": 6, "test": 6}
    for split, n in counts.items():
        for i in range(n):
            text = " ".join(rng.choice(WORDS) for _ in range(rng.randint(3, 6)))
            dur = rng.uniform(1.5, 3.0)
            wav = (nrng.standard_normal(int(dur * 16000)) * 0.05).astype("float32")
            rel = f"wav16k/{split}/{split}_{i:03d}.wav"
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            sf.write(root / rel, wav, 16000, subtype="PCM_16")
            region = ["North", "Central", "South"][i % 3]
            rows.append({"utt_id": f"{split}_{i:03d}", "split": split, "wav": rel, "duration": f"{len(wav) / 16000:.3f}",
                         "orig_sr": 16000, "text_raw": text.capitalize(), "region": region, "province_code": 11,
                         "province_name": "X", "gender": 0, "region_id": i % 3, "text_clean": text, "text_norm": text,
                         "has_digit": 0, "norm_changed": 0})
    write_tsv(root / "manifest.tsv", rows, FULL_FIELDS)
    model = train_bpe([r["text_norm"] for r in rows if r["split"] == "train"], tmp_path / "tok" / "bpe", vocab_size=40)
    return {"root": root, "tokenizer": model, "rows": rows}
