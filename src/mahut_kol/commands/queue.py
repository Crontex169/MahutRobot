"""
Kapsama garantili, devam ettirilebilir komut kuyruğu ve held-out kümesi.

Dengeli kuyruk:
- Her (manipüle nesne × yön) ve (referans nesne × yön) hücresi dengeli görülür.
- Bir sonraki komut, bu iki hücrenin sayaçları toplamı en küçük olan geçerli komuttur;
  eşitlikler (seed, ilerleme) ile belirlenen karıştırmayla bozulur -> sıra karışık ama
  tamamen tekrarlanabilir. Aynı komutun tekrarını azaltmak için komut sayacı da küçük
  ağırlıkla skora eklenir; sürekli başarısız olan komutlar (scripted) geri plana itilir.
- Not: TOP için 9 manipüle hücresine karşılık yalnızca 6 referans hücresi vardır (prizma
  referans olamaz). Her komut iki aileye de 1 ekler; bu yüzden TOP'ta iki aile aynı anda
  diğer yönlerle eşitlenemez. Algoritma her yön içinde hücreleri ±1 dengede tutar; TOP'un
  manipüle hücreleri biraz az, referans hücreleri biraz fazla görülür (900 komutta ~17 / ~25,
  diğer yönlerde 20-21).
- Sayaçlar yalnızca `commit` ile (bölüm kaydedildiğinde) artar; atılan bölümler kapsamayı
  bozmaz. Durum JSON dosyasına atomik yazılır (geçici dosya + rename), asla silinmez.

Held-out:
- 336 geçerli komuttan 30 tanesi (yön başına 6) değerlendirme için ayrılır. Hiçbir
  (manipüle×yön) ya da (referans×yön) hücresinden birden fazla seçilmez; böylece her hücrenin
  eğitimde en az 4 komutu kalır. Liste sabit JSON'dur (configs/commands/heldout_commands.json).
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

from mahut_kol.commands.schema import Command, all_valid_commands
from mahut_kol.envs.multi_shape_task import RELATIONS

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_HELDOUT_PATH = _REPO_ROOT / "configs" / "commands" / "heldout_commands.json"
HELDOUT_PER_RELATION = 6
HELDOUT_SEED = 20261107

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Held-out
# ---------------------------------------------------------------------------


def select_heldout(
    seed: int = HELDOUT_SEED,
    per_relation: int = HELDOUT_PER_RELATION,
    max_per_object: int = 4,
) -> List[Command]:
    """Kapsamayı bozmayacak şekilde held-out komutları seçer (deterministik)."""
    cmds = all_valid_commands()
    rng = random.Random(seed)
    rng.shuffle(cmds)
    used_m, used_r = set(), set()
    per_rel: Counter = Counter()
    per_m: Counter = Counter()
    per_r: Counter = Counter()
    out: List[Command] = []
    for c in cmds:
        if per_rel[c.relation] >= per_relation:
            continue
        if (c.manipulated_object, c.relation) in used_m or (c.reference_object, c.relation) in used_r:
            continue
        if per_m[c.manipulated_object] >= max_per_object or per_r[c.reference_object] >= max_per_object:
            continue
        out.append(c)
        used_m.add((c.manipulated_object, c.relation))
        used_r.add((c.reference_object, c.relation))
        per_rel[c.relation] += 1
        per_m[c.manipulated_object] += 1
        per_r[c.reference_object] += 1
    if len(out) != per_relation * len(RELATIONS):
        raise RuntimeError(f"Held-out seçimi tamamlanamadı: {len(out)}")
    return sorted(out, key=lambda c: (RELATIONS.index(c.relation), c.key))


def write_heldout(path: PathLike = DEFAULT_HELDOUT_PATH, seed: int = HELDOUT_SEED) -> Path:
    """Held-out JSON'unu yazar. Var olan dosyanın üzerine YAZMAZ."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Held-out dosyası zaten var, üzerine yazılmaz: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmds = select_heldout(seed)
    data = {
        "description": "Değerlendirme için ayrılan komutlar; kayıtta atlanır (--include-heldout ile açılır).",
        "seed": seed,
        "count": len(cmds),
        "commands": [c.to_dict() for c in cmds],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_heldout(path: PathLike = DEFAULT_HELDOUT_PATH) -> List[Command]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Held-out dosyası yok: {path}. Üretmek için: python3 scripts/make_heldout_split.py"
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Command(d["manipulated_object"], d["reference_object"], d["relation"]) for d in data["commands"]]


# ---------------------------------------------------------------------------
# Dengeli kuyruk
# ---------------------------------------------------------------------------


class BalancedCommandQueue:
    """
    Kullanım:
        q = BalancedCommandQueue(state_path, seed=0)
        cmd = q.peek()          # sıradaki komut (commit edilene kadar aynı kalır)
        ... bölüm kaydedildi ...
        q.commit(cmd)           # sayaçları güncelle + durumu diske yaz
        ... ya da başarısız oldu ...
        q.report_failure(cmd)   # komutu biraz geri plana it
    """

    STATE_VERSION = 1
    COMBO_WEIGHT = 0.25
    FAILURE_WEIGHT = 0.5

    def __init__(
        self,
        state_path: Optional[PathLike] = None,
        seed: int = 0,
        include_heldout: bool = False,
        heldout: Optional[Sequence[Command]] = None,
        heldout_path: PathLike = DEFAULT_HELDOUT_PATH,
    ):
        self.state_path = Path(state_path) if state_path is not None else None
        self.seed = int(seed)
        self.include_heldout = bool(include_heldout)
        if heldout is None:
            heldout = load_heldout(heldout_path)
        self.heldout_keys = {c.key for c in heldout}
        self.commands = [
            c for c in all_valid_commands() if self.include_heldout or c.key not in self.heldout_keys
        ]
        self.cell_m: Counter = Counter()
        self.cell_r: Counter = Counter()
        self.combo: Counter = Counter()
        self.failures: Counter = Counter()
        self.n_committed = 0
        self.n_failures = 0
        if self.state_path is not None and self.state_path.exists():
            self._load()

    # -- durum -------------------------------------------------------------
    def _load(self) -> None:
        data = json.loads(self.state_path.read_text(encoding="utf-8"))
        if data.get("version") != self.STATE_VERSION:
            raise ValueError(f"Kuyruk durum sürümü uyumsuz: {data.get('version')}")
        if data["seed"] != self.seed or data["include_heldout"] != self.include_heldout:
            raise ValueError(
                f"Kuyruk durum dosyası farklı ayarlarla oluşturulmuş (seed={data['seed']}, "
                f"include_heldout={data['include_heldout']}); {self.state_path}"
            )
        self.combo = Counter(data["combo_counts"])
        self.failures = Counter(data.get("failures", {}))
        self.n_committed = int(data["n_committed"])
        self.n_failures = int(data.get("n_failures", 0))
        for key, n in self.combo.items():
            c = Command.from_key(key)
            self.cell_m[(c.manipulated_object, c.relation)] += n
            self.cell_r[(c.reference_object, c.relation)] += n

    def _save(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": self.STATE_VERSION,
            "seed": self.seed,
            "include_heldout": self.include_heldout,
            "n_committed": self.n_committed,
            "n_failures": self.n_failures,
            "combo_counts": dict(sorted(self.combo.items())),
            "failures": dict(sorted(self.failures.items())),
        }
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_path)

    # -- seçim -------------------------------------------------------------
    def score(self, c: Command) -> float:
        return (
            self.cell_m[(c.manipulated_object, c.relation)]
            + self.cell_r[(c.reference_object, c.relation)]
            + self.COMBO_WEIGHT * self.combo[c.key]
            + self.FAILURE_WEIGHT * self.failures[c.key]
        )

    def peek(self) -> Command:
        rng = random.Random(f"{self.seed}:{self.n_committed}:{self.n_failures}")
        order = list(range(len(self.commands)))
        rng.shuffle(order)
        best_i = min(order, key=lambda i: self.score(self.commands[i]))  # min kararlı: ilk en küçük
        return self.commands[best_i]

    def commit(self, cmd: Command) -> None:
        if not self.include_heldout and cmd.key in self.heldout_keys:
            raise ValueError(f"Held-out komut kayda eklenemez: {cmd.key}")
        self.combo[cmd.key] += 1
        self.cell_m[(cmd.manipulated_object, cmd.relation)] += 1
        self.cell_r[(cmd.reference_object, cmd.relation)] += 1
        self.n_committed += 1
        self._save()

    def report_failure(self, cmd: Command) -> None:
        self.failures[cmd.key] += 1
        self.n_failures += 1
        self._save()

    def take(self, n: int) -> List[Command]:
        """n komutu sırayla üretip commit eder (simülasyon/test için)."""
        out = []
        for _ in range(n):
            c = self.peek()
            self.commit(c)
            out.append(c)
        return out

    # -- rapor -------------------------------------------------------------
    def coverage(self) -> Dict[str, Dict[Tuple[str, str], int]]:
        return {"manipulated_x_relation": dict(self.cell_m), "reference_x_relation": dict(self.cell_r)}


def coverage_tables(commands: Iterable[Command]) -> Tuple[Counter, Counter]:
    """(manipüle×yön) ve (referans×yön) sayaçları."""
    cm, cr = Counter(), Counter()
    for c in commands:
        cm[(c.manipulated_object, c.relation)] += 1
        cr[(c.reference_object, c.relation)] += 1
    return cm, cr
