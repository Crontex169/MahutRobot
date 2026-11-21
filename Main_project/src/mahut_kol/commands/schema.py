"""
Komut şeması: CommandParser JSON'u <-> ortam reset(options) <-> İngilizce talimat <-> slotlar.

Akış:  Türkçe ses/metin -> CommandParser JSON -> Command -> reset(options) + İngilizce talimat.
Eğitime giren metin her zaman sabit İngilizce şablondur; ham Türkçe/STT metni girmez.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from mahut_kol.envs.multi_shape_task import (
    DEFAULT_NUM_DISTRACTORS,
    OBJECT_IDS,
    RELATIONS,
    is_valid_command,
    validate_command,
)
from mahut_kol.envs.shape_objects import OBJECT_COLORS, OBJECT_SHAPES

COLORS: Tuple[str, ...] = tuple(OBJECT_COLORS.keys())  # ("red", "green", "blue")
SHAPES: Tuple[str, ...] = tuple(OBJECT_SHAPES)  # ("cube", "cylinder", "triangular_prism")
SLOT_NAMES = ("manipulated_color", "manipulated_shape", "reference_color", "reference_shape", "relation")


def split_object_id(object_id: str) -> Tuple[str, str]:
    """'blue_triangular_prism' -> ('blue', 'triangular_prism')."""
    color, shape = object_id.split("_", 1)
    if color not in COLORS or shape not in SHAPES:
        raise ValueError(f"Bilinmeyen nesne kimliği: '{object_id}'")
    return color, shape


@dataclass(frozen=True)
class Command:
    """Tek bir pick-and-place komutu (CommandParser alan adlarıyla)."""

    manipulated_object: str
    reference_object: str
    relation: str

    def __post_init__(self):
        object.__setattr__(self, "relation", self.relation.upper())
        validate_command(self.manipulated_object, self.reference_object, self.relation)

    # -- dönüşümler ---------------------------------------------------------
    @property
    def key(self) -> str:
        return f"{self.manipulated_object}|{self.reference_object}|{self.relation}"

    @classmethod
    def from_key(cls, key: str) -> "Command":
        m, r, rel = key.split("|")
        return cls(m, r, rel)

    @property
    def target_slot(self) -> str:
        return f"{self.reference_object}_{self.relation}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": "pick_and_place",
            "manipulated_object": self.manipulated_object,
            "reference_object": self.reference_object,
            "relation": self.relation,
            "target_slot": self.target_slot,
        }

    @classmethod
    def from_parser_json(cls, data: Mapping[str, Any]) -> "Command":
        """
        CommandParser.parse() çıktısından Command üretir. Geçersiz/eksik/onay bekleyen
        komutlarda ValueError (Türkçe mesaj) fırlatır; bu durumda bölüm başlatılmamalıdır.
        """
        if not data.get("valid", False):
            reason = data.get("error") or "komut geçersiz"
            if data.get("needs_confirmation"):
                reason = f"onay gerekli: {data.get('confirmation_prompt') or reason}"
            raise ValueError(f"Komut kabul edilmedi ({reason})")
        if data.get("action") != "pick_and_place":
            raise ValueError(
                f"Yalnızca 'pick_and_place' komutları kaydedilir, gelen eylem: '{data.get('action')}'"
            )
        for k in ("manipulated_object", "reference_object", "relation"):
            if not data.get(k):
                raise ValueError(f"Komutta '{k}' alanı eksik")
        return cls(data["manipulated_object"], data["reference_object"], data["relation"])

    def to_reset_options(
        self, num_distractors: int = DEFAULT_NUM_DISTRACTORS, **extra: Any
    ) -> Dict[str, Any]:
        """MultiShapeRelationalTask.reset(options=...) sözlüğü."""
        opts: Dict[str, Any] = {
            "manipulated_object": self.manipulated_object,
            "reference_object": self.reference_object,
            "relation": self.relation,
            "num_distractors": num_distractors,
        }
        opts.update(extra)
        return opts


# ---------------------------------------------------------------------------
# İngilizce talimat şablonları
# ---------------------------------------------------------------------------

# Yön ifadeleri: ilk eleman kanonik.
RELATION_PHRASES: Dict[str, Tuple[str, ...]] = {
    "FRONT": ("in front of",),
    "BACK": ("behind",),
    "LEFT": ("to the left of", "on the left side of"),
    "RIGHT": ("to the right of", "on the right side of"),
    "TOP": ("on top of",),
}

# Cümle kalıpları: ilk eleman kanonik.
INSTRUCTION_TEMPLATES: Tuple[str, ...] = (
    "pick the {m} and place it {rel} the {r}",
    "put the {m} {rel} the {r}",
    "place the {m} {rel} the {r}",
    "move the {m} {rel} the {r}",
    "pick up the {m} and put it {rel} the {r}",
)


def object_phrase(object_id: str) -> str:
    """'blue_triangular_prism' -> 'blue triangular prism'."""
    color, shape = split_object_id(object_id)
    return f"{color} {shape.replace('_', ' ')}"


def all_instructions(cmd: Command) -> List[str]:
    """Komut için tüm paraphrase'ler (sabit sıra, kanonik ilk)."""
    out = []
    for tpl in INSTRUCTION_TEMPLATES:
        for rel in RELATION_PHRASES[cmd.relation]:
            out.append(tpl.format(m=object_phrase(cmd.manipulated_object), rel=rel, r=object_phrase(cmd.reference_object)))
    return out


def to_english_instruction(
    cmd: Command,
    seed: Optional[int] = None,
    paraphrase: bool = True,
) -> str:
    """
    Sabit İngilizce talimat. paraphrase=False ya da seed=None -> kanonik cümle.
    Paraphrase seçimi (seed, komut) ile deterministiktir: aynı seed aynı cümleyi verir.
    """
    options = all_instructions(cmd)
    if not paraphrase or seed is None:
        return options[0]
    rng = random.Random(f"{seed}:{cmd.key}")
    return options[rng.randrange(len(options))]


# ---------------------------------------------------------------------------
# Sayısal slotlar
# ---------------------------------------------------------------------------


def to_slots(cmd: Command) -> np.ndarray:
    """
    5 sayısal slot (int64): [manipüle renk, manipüle şekil, referans renk, referans şekil, yön].
    İndeksler: renk COLORS, şekil SHAPES, yön RELATIONS sırasına göre.
    """
    mc, ms = split_object_id(cmd.manipulated_object)
    rc, rs = split_object_id(cmd.reference_object)
    return np.array(
        [COLORS.index(mc), SHAPES.index(ms), COLORS.index(rc), SHAPES.index(rs), RELATIONS.index(cmd.relation)],
        dtype=np.int64,
    )


def from_slots(slots) -> Command:
    s = [int(v) for v in slots]
    return Command(f"{COLORS[s[0]]}_{SHAPES[s[1]]}", f"{COLORS[s[2]]}_{SHAPES[s[3]]}", RELATIONS[s[4]])


def all_valid_commands() -> List[Command]:
    """336 geçerli komut (sabit sıra)."""
    return [
        Command(m, r, rel)
        for m in OBJECT_IDS
        for r in OBJECT_IDS
        for rel in RELATIONS
        if is_valid_command(m, r, rel)
    ]
