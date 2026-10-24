from pathlib import Path

import pytest
import yaml

from teda.utils.config import deep_merge, load_config, parse_override, path_exists

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def cfg_dir(tmp_path):
    (tmp_path / "exp").mkdir()
    (tmp_path / "base.yaml").write_text("a: {b: 1, c: [1, 2]}\nd: 3\n", encoding="utf-8")
    (tmp_path / "exp" / "x.yaml").write_text("a: {c: [9]}\n", encoding="utf-8")
    return tmp_path


def test_merge_replaces_lists_and_merges_dicts():
    assert deep_merge({"a": {"b": 1, "c": [1, 2]}}, {"a": {"c": [9]}}) == {"a": {"b": 1, "c": [9]}}


def test_load_config_exp_name_and_overrides(cfg_dir):
    cfg = load_config(cfg_dir / "base.yaml", "x", ["a.b=5", "d=null", "a.c=[0,1]"])
    assert cfg.exp_name == "x" and cfg.a.b == 5 and cfg.d is None and cfg.a.c == [0, 1]
    assert parse_override("t.k=true") == ("t.k", True) and parse_override("t.k=1.5e-4") == ("t.k", 1.5e-4)


def test_override_typos_are_rejected_with_a_suggestion(cfg_dir):
    with pytest.raises(KeyError, match="a.b"):
        load_config(cfg_dir / "base.yaml", None, ["a.bb=5"])  # silently ignored overrides waste whole training runs
    cfg = load_config(cfg_dir / "base.yaml", None, ["+a.new=7"])  # explicit '+' adds a brand-new key
    assert cfg.a.new == 7 and path_exists(cfg, "a.new") and not path_exists(cfg, "a.zzz")


def test_unknown_experiment_is_an_error(cfg_dir):
    with pytest.raises(FileNotFoundError):
        load_config(cfg_dir / "base.yaml", "nope")


def test_repo_configs_load_and_have_the_agreed_values():
    cfg = load_config(ROOT / "configs" / "base.yaml", "e00_baseline")
    assert cfg.exp_name == "e00_baseline" and cfg.data.vocab_size == 1024
    assert cfg.data.regions == ["North", "Central", "South"]
    assert cfg.encoder.overrides.dynamic_chunk_sizes == [-1, -1, 64, 128, 256]
    assert cfg.train.amp == "fp16" and cfg.encoder.overrides.encoder_layerdrop == 0.0


def test_sweep_file_is_consistent_with_the_base_config():
    sw = yaml.safe_load(open(ROOT / "configs" / "sweep" / "freeze_search.yaml", encoding="utf-8"))
    base = load_config(ROOT / "configs" / "base.yaml", sw["base_exp"])
    names = [r["name"] for r in sw["runs"]]
    assert len(names) == len(set(names)), "run names must be unique (they are output folder names)"
    for run in sw["runs"]:
        for key in {**sw.get("common_set", {}), **run["set"]}:
            assert path_exists(base, key), f"sweep run '{run['name']}': unknown config key '{key}'"
        layers = run["set"].get("encoder.freeze_layers", [])
        assert all(isinstance(i, int) and 0 <= i < 12 for i in layers), run["name"]
        assert len(layers) == len(set(layers))
