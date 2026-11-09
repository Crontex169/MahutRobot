"""Türev ACT veri kümesi: kaynak değişmez, one-hot sütunu eklenir, var olan hedef ezilmez."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("make_act_dataset", ROOT / "scripts" / "make_act_dataset.py")
mad = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mad)


def _make_src(root: Path):
    (root / "meta").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "videos" / "cam").mkdir(parents=True)
    (root / "videos" / "cam" / "a.mp4").write_bytes(b"video")
    (root / "meta" / "info.json").write_text(json.dumps({"features": {"command.slots": {"dtype": "int64", "shape": [5]}}}))
    (root / "meta" / "stats.json").write_text(json.dumps({"command.slots": {}}))
    slots = np.array([[0, 1, 2, 0, 4], [1, 1, 1, 1, 0], [2, 2, 0, 0, 3]], dtype=np.int64)
    col = pa.FixedSizeListArray.from_arrays(pa.array(slots.reshape(-1)), 5)
    pq.write_table(pa.table({"command.slots": col}), root / "data" / "chunk-000" / "file-000.parquet")


def test_build_adds_onehot_and_keeps_source(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_src(src)
    before = (src / "meta" / "info.json").read_text()
    res = mad.build(src, dst)
    assert res["frames"] == 3
    t = pq.read_table(dst / "data" / "chunk-000" / "file-000.parquet")
    env = np.asarray(t.column("observation.environment_state").to_pylist())
    assert env.shape == (3, 17) and (env.sum(1) == 5).all()
    info = json.loads((dst / "meta" / "info.json").read_text())
    assert info["features"]["observation.environment_state"]["shape"] == [17]
    assert "observation.environment_state" in json.loads((dst / "meta" / "stats.json").read_text())
    assert (dst / "videos" / "cam" / "a.mp4").read_bytes() == b"video"
    assert (src / "meta" / "info.json").read_text() == before  # kaynak değişmedi


def test_existing_destination_is_never_overwritten(tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _make_src(src)
    dst.mkdir()
    with pytest.raises(FileExistsError):
        mad.build(src, dst)
