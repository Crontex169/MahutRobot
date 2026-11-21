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

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein


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
    # Not: "al" kısa olduğu için yalnızca emir/zarf-fiil çekimleri kabul edilir;
    # aksi halde "alt", "alan", "alana" gibi kelimeler yanlışlıkla pick sayılır.
    (r"\b(al|alın|alin|alınız|aliniz|alıp|alip|alır|alir|alsana|alabilir[a-zıüöçşğ]*|kaldır[a-zıüöçşğ]*|kaldir[a-zıüöçşğ]*|tut[a-zıüöçşğ]*)\b", "pick"),
]


# ---------------------------------------------------------------------------
# Bulanık (fuzzy) eşleme: STT yazım hatalarına karşı slot bazlı yedek çözümleme
# ---------------------------------------------------------------------------

_TR_FOLD = str.maketrans("çğıöşüâîû", "cgiosuaiu")


def fold_turkish(text: str) -> str:
    """Türkçe karakterleri ASCII karşılıklarına indirger (ç->c, ğ->g, ı->i, ö->o, ş->s, ü->u)."""
    return text.translate(_TR_FOLD)


# (kök, kategori, değer, düzeltmede yazılacak kanonik biçim)
# Kanonik biçimler mevcut kesin regex kalıplarının kesin olarak tanıdığı biçimlerdir.
FUZZY_LEXICON: List[Tuple[str, str, str, str]] = [
    ("kırmızı", "color", "red", "kırmızı"),
    ("yeşil", "color", "green", "yeşil"),
    ("mavi", "color", "blue", "mavi"),
    ("küp", "shape", "cube", "küp"),
    ("silindir", "shape", "cylinder", "silindir"),
    ("prizma", "shape", "triangular_prism", "prizma"),
    ("üçgen", "shape", "triangular_prism", "üçgen"),
    ("ön", "relation", "FRONT", "önüne"),
    ("arka", "relation", "BACK", "arkasına"),
    ("sol", "relation", "LEFT", "soluna"),
    ("sağ", "relation", "RIGHT", "sağına"),
    ("üst", "relation", "TOP", "üstüne"),
    ("üzer", "relation", "TOP", "üzerine"),
    ("koy", "action", "place", "koy"),
    ("bırak", "action", "place", "bırak"),
    ("yerleştir", "action", "place", "yerleştir"),
    ("taşı", "action", "place", "taşı"),
    ("it", "action", "push", "it"),
    ("itele", "action", "push", "itele"),
    ("ittir", "action", "push", "ittir"),
    ("kaydır", "action", "push", "kaydır"),
    ("al", "action", "pick", "al"),
    ("kaldır", "action", "pick", "kaldır"),
    ("tut", "action", "pick", "tut"),
]

# Kategori bazlı izinli ekler (ASCII'ye indirgenmiş; ünlü uyumu indirgemeyle 2 biçime düşer).
_NOUN_SUFFIXES = (
    "", "i", "u", "a", "e", "yi", "yu", "ya", "ye", "in", "un", "nin", "nun",
    "de", "da", "te", "ta", "den", "dan", "ten", "tan", "le", "la", "yle", "yla",
)
# Yön kelimelerinde yalın "u/i/a/e" ekleri bilinçli olarak YOK: "onu", "ona" gibi
# zamirler "ön" köküne düşmesin. Yön için iyelik + hal ekleri beklenir.
_RELATION_SUFFIXES = (
    "", "si", "su", "une", "una", "ine", "ina", "unde", "unda", "inde", "inda",
    "na", "ne", "nda", "nde", "ndaki", "ndeki",
    "undaki", "undeki", "indaki", "indeki", "sina", "sine", "sinda", "sinde",
    "sindaki", "sindeki",
)
_ACTION_SUFFIXES = ("", "in", "un", "iniz", "unuz", "ip", "up", "sana", "sene", "abilir", "ebilir")

FUZZY_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    "color": _NOUN_SUFFIXES,
    "shape": _NOUN_SUFFIXES,
    "relation": _RELATION_SUFFIXES,
    "action": _ACTION_SUFFIXES,
}

# Türkçe gösterim (onay sorusu üretimi için)
COLOR_TR = {"red": "kırmızı", "green": "yeşil", "blue": "mavi"}
SHAPE_TR = {"cube": "küp", "cylinder": "silindir", "triangular_prism": "üçgen prizma"}
RELATION_TR = {"FRONT": "önü", "BACK": "arkası", "LEFT": "solu", "RIGHT": "sağı", "TOP": "üstü"}

_VOWELS = "aeıioöuü"


def _last_vowel(word: str) -> str:
    for ch in reversed(word):
        if ch in _VOWELS:
            return ch
    return "e"


def _question_particle(phrase: str) -> str:
    """Ünlü uyumuna göre soru eki (mı/mi/mu/mü)."""
    v = _last_vowel(phrase)
    return {"a": "mı", "ı": "mı", "e": "mi", "i": "mi", "o": "mu", "u": "mu", "ö": "mü", "ü": "mü"}[v]


def _genitive(phrase: str) -> str:
    """Basit Türkçe ilgi hali eki (küp -> küpün, prizma -> prizmanın)."""
    v = _last_vowel(phrase)
    vowel = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[v]
    return phrase + ("n" if phrase[-1] in _VOWELS else "") + vowel + "n"


def _object_tr(color: str, shape: str) -> str:
    return f"{COLOR_TR[color]} {SHAPE_TR[shape]}"


class CommandParser:
    """Türkçe robot sesli komutlarını yapılandırılmış JSON nesnesine dönüştüren parser."""

    def __init__(
        self,
        enable_fuzzy: bool = True,
        accept_threshold: float = 85.0,
        reject_threshold: float = 75.0,
        ambiguity_margin: float = 5.0,
    ):
        """
        enable_fuzzy: Kesin eşleşme başarısız olduğunda bulanık eşleme yedeğini kullan.
        accept_threshold: Bu skorun (0-100) üstündeki bulanık eşleşmeler onaysız kabul edilir.
        reject_threshold: Bu skorun altındaki eşleşmeler reddedilir. Aradaki skorlar
            komutu valid=False + needs_confirmation=True yapar.
        ambiguity_margin: Farklı anlamlara giden iki aday bu kadar yakınsa token belirsiz sayılır.
        """
        self.enable_fuzzy = enable_fuzzy
        self.accept_threshold = accept_threshold
        self.reject_threshold = reject_threshold
        self.ambiguity_margin = ambiguity_margin

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

    # ------------------------------------------------------------------
    # Bulanık eşleme yardımcıları
    # ------------------------------------------------------------------

    @staticmethod
    def _token_is_known(token: str) -> bool:
        """Token mevcut kesin kalıplardan biriyle tam olarak eşleşiyor mu?"""
        if token in COLOR_MAP:
            return True
        for pattern, _ in SHAPE_PATTERNS + RELATION_PATTERNS + ACTION_PATTERNS:
            if re.fullmatch(pattern, token):
                return True
        return False

    @staticmethod
    def _stem_score(body: str, body_folded: str, stem: str, stem_folded: str) -> Optional[float]:
        """
        Kök adayı ile token gövdesinin benzerlik skoru (0-100) ya da uygun değilse None.

        - Kısa kökler (<=3 harf: al, it, ön, sol, sağ, küp, koy...): yalnızca ASCII
          yazımı farkına izin verilir, harf hatasına izin YOK. Ayrıca token'da Türkçe
          karakter varsa kökle birebir aynı olmalı ("köy" -> "koy" değil).
        - 4-5 harfli kökler: en fazla 1, 6+ harfli kökler: en fazla 2 düzenleme.
        - İlk harf (ASCII indirgenmiş) aynı olmalı; Whisper hataları genelde baştan değil.
        """
        if not body_folded or body_folded[0] != stem_folded[0]:
            return None
        if len(stem_folded) <= 3:
            if len(body) != len(stem):
                return None
            for b, st in zip(body, stem):
                if b != st and not (b.isascii() and fold_turkish(st) == b):
                    return None
            return 100.0
        max_edits = 1 if len(stem_folded) <= 5 else 2
        if Levenshtein.distance(body_folded, stem_folded) > max_edits:
            return None
        return float(fuzz.ratio(body_folded, stem_folded))

    def match_token(self, token: str) -> Optional[Dict[str, Any]]:
        """
        Tek bir token'ı (ör. "marvi", "silindrin", "sagina") renk/şekil/yön/eylem
        sözlüğüne bulanık olarak eşler. Türkçe ekler önce ayrılır (kök + ek).

        Dönüş: {stem, category, value, surface, suffix, score, ambiguous, alternatives}
        veya eşik altında/eşleşme yoksa None.
        """
        token_folded = fold_turkish(token)
        best_per_value: Dict[Tuple[str, str], Dict[str, Any]] = {}

        for stem, category, value, surface in FUZZY_LEXICON:
            stem_folded = fold_turkish(stem)
            for suffix in FUZZY_SUFFIXES[category]:
                if suffix and not token_folded.endswith(suffix):
                    continue
                cut = len(token) - len(suffix)
                body, body_folded = token[:cut], token_folded[:cut]
                if not body:
                    continue
                # Kısa yön kökleri eksiz kabul edilmez ("on" = 10, "sol" nota vb.);
                # eksiz biçimleri zaten kesin regex yakalıyor.
                if category == "relation" and not suffix and len(stem_folded) <= 3:
                    continue
                score = self._stem_score(body, body_folded, stem, stem_folded)
                if score is None:
                    continue
                key = (category, value)
                prev = best_per_value.get(key)
                if prev is None or score > prev["score"]:
                    best_per_value[key] = {
                        "stem": stem,
                        "category": category,
                        "value": value,
                        "surface": surface,
                        "suffix": suffix,
                        "score": score,
                    }

        candidates = sorted(best_per_value.values(), key=lambda c: -c["score"])
        if not candidates or candidates[0]["score"] < self.reject_threshold:
            return None
        best = dict(candidates[0])
        rivals = [
            c for c in candidates[1:]
            if c["score"] >= self.reject_threshold
            and best["score"] - c["score"] <= self.ambiguity_margin
        ]
        best["ambiguous"] = bool(rivals)
        best["alternatives"] = [{"matched": c["stem"], "score": round(c["score"], 1)} for c in rivals]
        return best

    def _build_confirmation_prompt(
        self,
        weakest: Dict[str, Any],
        corrected_text: str,
        token_index: int,
        result: Dict[str, Any],
    ) -> str:
        """En düşük güvenli eşleşmeye göre kullanıcıya sorulacak soruyu üretir."""
        category = weakest["category"]
        if category in ("color", "shape"):
            # Token'ın hangi nesneye ait olduğunu karakter konumundan bul
            offset = sum(len(t) + 1 for t in corrected_text.split()[:token_index])
            for obj in self.extract_objects(corrected_text):
                if obj["start"] <= offset < obj["end"]:
                    phrase = _object_tr(obj["color"], obj["shape"])
                    break
            else:
                phrase = weakest["surface"]
        elif category == "relation" and result.get("details", {}).get("reference"):
            ref = result["details"]["reference"]
            phrase = f"{_genitive(_object_tr(ref['color'], ref['shape']))} {RELATION_TR[weakest['value']]}"
        elif category == "action":
            phrase = f"'{weakest['surface']}' eylemi"
        else:
            phrase = weakest["surface"]
        phrase = phrase[0].upper() + phrase[1:]
        return f"{phrase} {_question_particle(phrase)} demek istediniz?"

    def _fuzzy_parse(self, text: str, norm_text: str, exact_result: Dict[str, Any]) -> Dict[str, Any]:
        """Token bazlı bulanık düzeltme yapıp kesin çözümleyiciyi düzeltilmiş metinle tekrar çalıştırır."""
        tokens = norm_text.split()
        corrected: List[str] = []
        fuzzy_matches: List[Dict[str, Any]] = []
        weakest: Optional[Tuple[Dict[str, Any], int]] = None

        for idx, tok in enumerate(tokens):
            if len(tok) < 2 or self._token_is_known(tok):
                corrected.append(tok)
                continue
            m = self.match_token(tok)
            if m is None:
                corrected.append(tok)
                continue
            corrected.append(m["surface"])
            entry = {
                "token": tok,
                "matched": m["stem"],
                "category": m["category"],
                "value": m["value"],
                "score": round(m["score"], 1),
            }
            if m["ambiguous"]:
                entry["ambiguous"] = True
                entry["alternatives"] = m["alternatives"]
            fuzzy_matches.append(entry)
            # Onay gerektiren en zayıf eşleşmeyi takip et (belirsizler öncelikli)
            rank = (0 if m["ambiguous"] else 1, m["score"])
            if weakest is None or rank < (0 if weakest[0]["ambiguous"] else 1, weakest[0]["score"]):
                weakest = (m, idx)

        if not fuzzy_matches:
            return exact_result

        corrected_text = " ".join(corrected)
        result = self._parse_exact(corrected_text)
        result["raw_text"] = text
        result["normalized_text"] = norm_text
        details = result.get("details") or {}
        details["corrected_text"] = corrected_text
        details["fuzzy_matches"] = fuzzy_matches
        result["details"] = details

        if not result["valid"]:
            # Düzeltmeye rağmen çözülemedi: kesin çözümleyicinin hata mesajını koru
            return result

        min_score = min(fm["score"] for fm in fuzzy_matches)
        any_ambiguous = any(fm.get("ambiguous") for fm in fuzzy_matches)
        result["confidence"] = round(result["confidence"] * min_score / 100.0, 3)

        if any_ambiguous or min_score < self.accept_threshold:
            assert weakest is not None
            prompt = self._build_confirmation_prompt(weakest[0], corrected_text, weakest[1], result)
            weak = weakest[0]
            reason = "belirsiz" if weak["ambiguous"] else f"düşük güvenli (%{weak['score']:.0f})"
            result["valid"] = False
            result["needs_confirmation"] = True
            result["confirmation_prompt"] = prompt
            result["error"] = (
                f"'{tokens[weakest[1]]}' kelimesi '{weak['stem']}' olarak {reason} eşleşti; onay gerekli. {prompt}"
            )
        return result

    # ------------------------------------------------------------------
    # Ana giriş noktası
    # ------------------------------------------------------------------

    def parse(self, text: str) -> Dict[str, Any]:
        """
        Kullanıcıdan gelen ham metni yapılandırılmış JSON robot komutuna çevirir.

        Önce kesin (regex) çözümleme denenir. Başarısız olursa ve enable_fuzzy açıksa
        token bazlı bulanık eşleme ile STT yazım hataları ("marvi", "kırmzı") düzeltilir.
        Bulanık eşleşmeler details["fuzzy_matches"] içinde raporlanır; düşük güvenli veya
        belirsiz eşleşmelerde valid=False ve needs_confirmation=True döner.
        """
        norm_text = self.normalize_text(text)
        result = self._parse_exact(text)
        if not norm_text or not self.enable_fuzzy:
            return result
        if not result["valid"]:
            return self._fuzzy_parse(text, norm_text, result)

        # Kesin çözüm geçerli olsa bile, tanınmayan bir token bulanık olarak bir slota
        # eşleşiyorsa ve farklı bir komut ortaya çıkıyorsa kesin sonuç eksik okumadır.
        # Örn. "kırmızı küpü al yeşil silindirin üstne koy" kesin çözümde yön kaçtığı
        # için sessizce "pick red_cube" olurdu.
        fuzzy_result = self._fuzzy_parse(text, norm_text, result)
        if fuzzy_result is result:
            return result
        slots = ("action", "manipulated_object", "reference_object", "relation")
        if any(fuzzy_result[k] != result[k] for k in slots) and (
            fuzzy_result["valid"] or fuzzy_result["needs_confirmation"]
        ):
            return fuzzy_result
        return result

    def _parse_exact(self, text: str) -> Dict[str, Any]:
        """Yalnızca kesin regex/COLOR_MAP eşleşmeleriyle çözümleme (bulanık eşleme yok)."""
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
            "needs_confirmation": False,
            "confirmation_prompt": None,
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
