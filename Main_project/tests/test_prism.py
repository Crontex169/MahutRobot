"""Üçgen prizma meshi: geometri, depodaki OBJ ile üretecin uyumu. Fizik: test_shape_objects."""

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from mahut_kol.envs import prism_mesh as gpm

ROOT = Path(__file__).resolve().parents[1]

OBJ = ROOT / "assets" / "objects" / "triangular_prism.obj"


@pytest.mark.parametrize("a,h", [(0.035, 0.035), (0.02, 0.05)])
def test_geometry(a, h):
    mesh = trimesh.Trimesh(gpm.prism_vertices(a, h), gpm.FACES, process=False)
    assert mesh.vertices.shape == (6, 3) and mesh.faces.shape == (8, 3)
    assert mesh.is_watertight and mesh.is_convex and mesh.is_winding_consistent
    assert mesh.volume > 0  # pozitif hacim = normaller dışa bakıyor
    assert mesh.volume == pytest.approx(math.sqrt(3) / 4 * a * a * h, rel=1e-9)
    np.testing.assert_allclose(mesh.center_mass, 0.0, atol=1e-9)
    np.testing.assert_allclose(mesh.extents, [a, math.sqrt(3) / 2 * a, h], atol=1e-12)


def test_committed_obj_matches_generator():
    assert OBJ.read_text(encoding="utf-8") == gpm.prism_obj_text()
    mesh = trimesh.load(OBJ, process=False)
    assert len(mesh.vertices) == 6 and mesh.is_watertight


@pytest.mark.parametrize("a,h", [(0.0, 0.035), (-0.035, 0.035), (0.035, 0.0)])
def test_rejects_non_positive(a, h):
    with pytest.raises(ValueError):
        gpm.prism_vertices(a, h)


def test_cli_writes_file(tmp_path):
    out = tmp_path / "sub" / "p.obj"
    gpm.main(["--out", str(out)])
    assert out.read_text(encoding="utf-8") == gpm.prism_obj_text()
