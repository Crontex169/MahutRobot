"""17 boyutlu one-hot komut kodlaması (Modal hesabı gerektirmez)."""

import numpy as np
import pytest

from mahut_kol.commands import all_valid_commands, to_slots
from mahut_kol.commands.onehot import ONEHOT_DIM, SLOT_SIZES, slots_to_onehot
from mahut_kol.commands.schema import COLORS, SHAPES
from mahut_kol.envs.multi_shape_task import RELATIONS


def test_block_sizes_match_schema():
    assert SLOT_SIZES == (len(COLORS), len(SHAPES), len(COLORS), len(SHAPES), len(RELATIONS))
    assert ONEHOT_DIM == 17


def test_each_block_has_exactly_one_hot():
    v = slots_to_onehot(to_slots(all_valid_commands()[0]))
    assert v.shape == (17,) and v.dtype == np.float32
    off = 0
    for size in SLOT_SIZES:
        assert v[off:off + size].sum() == 1.0
        off += size


def test_all_336_commands_unique_and_batch_equals_single():
    slots = np.stack([to_slots(c) for c in all_valid_commands()])
    batch = slots_to_onehot(slots)
    assert batch.shape == (336, 17)
    assert len({tuple(r) for r in batch}) == 336
    np.testing.assert_array_equal(batch[5], slots_to_onehot(slots[5]))


def test_known_vector():
    # [red=0, cube=0, blue=2, prism=2, RELATIONS[1]]
    v = slots_to_onehot([0, 0, 2, 2, 1])
    assert list(np.flatnonzero(v)) == [0, 3, 8, 11, 13]


def test_out_of_range_raises():
    with pytest.raises(ValueError):
        slots_to_onehot([3, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        slots_to_onehot([0, 0, 0, 0])
