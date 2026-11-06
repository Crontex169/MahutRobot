"""
Türkçe Doğal Dil Robot Komut Çözücü ve Normalizasyon Modülü.

Bu modül, mikrofondan veya STT motorundan gelen ham Türkçe talimatları
ManiSkill MultiShapeRelationalTask ortamının anlayabileceği yapılandırılmış
JSON komutlarına (slot extraction) dönüştürür.
"""

from __future__ import annotations

import re
import string
from typing import Any, Dict, List, Optional, Tuple


def turkish_lower(text: str) -> str:
    """
    Türkçe büyük harfleri küçük harflere hatasız dönüştürür.
    Python varsayılan .lower() metodunun 'İ' ve 'I' harflerini bozmasını önler.
    """
    mapping = {
        "İ": "i",
        "I": "ı",
    }
    for upper_char, lower_char in mapping.items():
        text = text.replace(upper_char, lower_char)
    return text.lower()


# ManiSkill / MahutKol nesne tanımları
COLOR_MAP = {
    "kırmızı": "red",
    "kirmizi": "red",
    "yeşil": "green",
    "yesil": "green",
    "mavi": "blue",
}

SHAPE_PATTERNS = [
    # (Regex pattern, İngilizce karşılık, öncelik)
    (r"\b(üçgen\s+prizma[a-zıüöçşğ]*|ucgen\s+prizma[a-zıüöçşğ]*)\b", "triangular_prism"),
    (r"\b(prizma[a-zıüöçşğ]*)\b", "triangular_prism"),
    (r"\b(üçgen[a-zıüöçşğ]*|ucgen[a-zıüöçşğ]*)\b", "triangular_prism"),
    (r"\b(silindir[a-zıüöçşğ]*)\b", "cylinder"),
    (r"\b(küp[a-zıüöçşğ]*|kup[a-zıüöçşğ]*)\b", "cube"),
]

RELATION_PATTERNS = [
    (r"\b(ön|on)(ü|üne|ünde|deki|ündeki)?\b", "FRONT"),
    (r"\b(arka)(sı|sına|sında|daki|sındaki)?\b", "BACK"),
    (r"\b(sol)(u|una|unda|daki|undaki)?\b", "LEFT"),
    (r"\b(sağ|sag)(ı|ına|ında|daki|i|ine|inde|ındaki|indeki)?\b", "RIGHT"),
    (r"\b(üst|ust|üzer|uzer|tepe)(ü|üne|ünde|i|ine|inde|si|sine|ündeki|indeki)?\b", "TOP"),
]

ACTION_PATTERNS = [
    # pick_and_place eylemleri: koy, bırak, yerleştir, taşı
    (r"\b(koy[a-zıüöçşğ]*|bırak[a-zıüöçşğ]*|birak[a-zıüöçşğ]*|yerleştir[a-zıüöçşğ]*|yerlestir[a-zıüöçşğ]*|taşı[a-zıüöçşğ]*|tasi[a-zıüöçşğ]*)\b", "place"),
    # push eylemleri: it, itele, kaydır, ittir
    (r"\b(it|itele[a-zıüöçşğ]*|kaydır[a-zıüöçşğ]*|kaydir[a-zıüöçşğ]*|ittir[a-zıüöçşğ]*)\b", "push"),
    # pick eylemleri: al, kaldır, tut
    (r"\b(al[a-zıüöçşğ]*|kaldır[a-zıüöçşğ]*|kaldir[a-zıüöçşğ]*|tut[a-zıüöçşğ]*)\b", "pick"),
]


class CommandParser:
    """Türkçe robot sesli komutlarını yapılandırılmış JSON nesnesine dönüştüren parser."""

    def __init__(self):
        pass

    def normalize_text(self, text: str) -> str:
        """Metni küçük harfe çevirir, noktalama işaretlerini temizler ve boşlukları düzenler."""
        if not text:
            return ""
        lowered = turkish_lower(text)
        # Noktalama işaretlerini boşlukla değiştir
        cleaned = re.sub(r"[^\w\s]", " ", lowered)
        # Çoklu boşlukları teke indir
        return re.sub(r"\s+", " ", cleaned).strip()

    def extract_objects(self, text: str) -> List[Dict[str, Any]]:
        """
        Metin içindeki nesneleri (renk + şekil) sırasıyla tespit eder.
        Dönen liste elemanları:
        {
            "color": "red",
            "shape": "cube",
            "object_id": "red_cube",
            "start": int,
            "end": int,
            "matched_text": str
        }
        """
        norm_text = self.normalize_text(text)
        found_objects = []

        # Renk ve şekil çiftlerini ara
        # Regex: (renk) + isteğe bağlı 1-2 aracı kelime + (şekil)
        color_regex = "|".join(re.escape(c) for c in COLOR_MAP.keys())

        # Tüm renk pozisyonlarını bul
        for color_match in re.finditer(rf"\b({color_regex})\b", norm_text):
            tr_color = color_match.group(1)
            en_color = COLOR_MAP[tr_color]
            c_start = color_match.start()
            c_end = color_match.end()

            # Rengin ardından (maksimum 35 karakter içinde) gelen şekilleri bul
            window_end = min(len(norm_text), c_end + 35)
            window = norm_text[c_start:window_end]

            # Pencere içinde renkten sonraki ilk şekli bul
            matches = []
            for pattern, en_shape in SHAPE_PATTERNS:
                for m in re.finditer(pattern, window):
                    # Şekil renkten sonra başlamalı
                    if m.start() >= (c_end - c_start):
                        matches.append((m.start(), m.end(), en_shape))

            if matches:
                # Başlangıç konumuna göre (en yakın olan), eşitse en uzun olana göre sırala
                matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
                best_start, best_end, best_shape = matches[0]

                obj_start = c_start
                obj_end = c_start + best_end
                obj_id = f"{en_color}_{best_shape}"
                matched_str = norm_text[obj_start:obj_end]

                found_objects.append({
                    "color": en_color,
                    "shape": best_shape,
                    "object_id": obj_id,
                    "start": obj_start,
                    "end": obj_end,
                    "matched_text": matched_str,
                })

        # Tekrarlayan veya örtüşen eşleşmeleri temizle
        unique_objects: List[Dict[str, Any]] = []
        for obj in found_objects:
            overlap = False
            for u in unique_objects:
                if not (obj["end"] <= u["start"] or obj["start"] >= u["end"]):
                    overlap = True
                    break
            if not overlap:
                unique_objects.append(obj)

        unique_objects.sort(key=lambda x: x["start"])
        return unique_objects

    def extract_relation(self, text: str) -> Optional[Tuple[str, int, int]]:
        """Metin içindeki uzamsal ilişkiyi (FRONT, BACK, LEFT, RIGHT, TOP) bulur."""
        norm_text = self.normalize_text(text)
        for pattern, rel in RELATION_PATTERNS:
            m = re.search(pattern, norm_text)
            if m:
                return rel, m.start(), m.end()
        return None

    def extract_actions(self, text: str) -> List[Tuple[str, int, int]]:
        """Metindeki eylem fiillerini bulur."""
        norm_text = self.normalize_text(text)
        actions = []
        for pattern, act in ACTION_PATTERNS:
            for m in re.finditer(pattern, norm_text):
                actions.append((act, m.start(), m.end()))
        actions.sort(key=lambda x: x[1])
        return actions

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Kullanıcıdan gelen ham metni yapılandırılmış JSON robot komutuna çevirir.
        """
        norm_text = self.normalize_text(text)

        result: Dict[str, Any] = {
            "raw_text": text,
            "normalized_text": norm_text,
            "action": None,
            "manipulated_object": None,
            "reference_object": None,
            "relation": None,
            "target_slot": None,
            "valid": False,
            "confidence": 0.0,
            "details": {},
            "error": None,
        }

        if not norm_text:
            result["error"] = "Boş girdi"
            return result

        # 1. Nesneleri çıkar
        objects = self.extract_objects(norm_text)
        # 2. Bağıl yönü çıkar
        relation_info = self.extract_relation(norm_text)
        relation = relation_info[0] if relation_info else None
        # 3. Eylemleri çıkar
        actions = self.extract_actions(norm_text)

        if not objects:
            result["error"] = "Metinde geçerli bir robot manipülasyon nesnesi bulunamadı"
            return result

        manipulated_obj = objects[0]["object_id"]
        reference_obj = objects[1]["object_id"] if len(objects) > 1 else None

        # Eylem ve amaç analizi
        action_names = [a[0] for a in actions]

        # 'al' ve 'koy/bırak' birlikte geçiyorsa veya hedef nesne + yön varsa bu pick_and_place dir
        if "place" in action_names or ("pick" in action_names and reference_obj and relation):
            action = "pick_and_place"
        elif "push" in action_names:
            action = "push"
        elif "pick" in action_names:
            action = "pick"
        else:
            action = None

        if action is None:
            result["error"] = "Metinde geçerli bir eylem fiili (al, koy, it, taşı, kaldır vb.) bulunamadı"
            return result

        # İlişki kontrolü
        # Eğer pick_and_place ise hem referans nesne hem yön şarttır
        if action == "pick_and_place":
            if not reference_obj:
                result["error"] = "Hedef referans nesne (örn. yeşil silindirin) belirtilmedi"
                return result
            if not relation:
                result["error"] = f"Hedef referans nesne ({reference_obj}) belirtildi ancak yön (sağına, soluna vb.) bulunamadı"
                return result

        # Sonuçları paketle
        target_slot = f"{reference_obj}_{relation}" if (reference_obj and relation) else None

        result.update({
            "action": action,
            "manipulated_object": manipulated_obj,
            "reference_object": reference_obj,
            "relation": relation,
            "target_slot": target_slot,
            "valid": True,
            "confidence": 1.0 if (manipulated_obj and (action != 'pick_and_place' or target_slot)) else 0.8,
            "details": {
                "manipulated": {
                    "color": objects[0]["color"],
                    "shape": objects[0]["shape"],
                    "object_id": manipulated_obj,
                },
                "reference": {
                    "color": objects[1]["color"],
                    "shape": objects[1]["shape"],
                    "object_id": reference_obj,
                } if reference_obj else None,
                "relation": relation,
                "all_objects": [o["object_id"] for o in objects],
                "action_raw": action_names,
            },
        })

        return result
