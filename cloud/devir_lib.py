"""
Eğitim devri (Data_Model) için SAF mantık: manifest, güvenli tar, özet doğrulama, çatallanma,
git denetimi ve kredi hesabı. Yalnızca standart kütüphane (+ isteğe bağlı pyyaml) kullanır;
Modal, torch, lerobot gerekmez, bu yüzden Modal hesabı olmadan test edilir.

Düzen (Data_Model/, git'e girmez):
    sabit/sabit_v<N>.tar                         bir kez indirilir, değişmez
    iterasyonlar/<hat>/iter_<NNNN>_step_<ADIM>_<kisi>.tar

Güvenlik duruşu: tar'lar güvenilmeyen girdi sayılır. Yalnızca beklenen yol desenleri ve düz dosya
türleri kabul edilir; pickle (.pt/.pth/.bin/.pkl), mutlak yol, '..', sembolik/sert bağ, cihaz
dosyası ve manifest'te olmayan üye reddedilir. Çıkarma elle ve hedef klasör içinde yapılır.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath

FORMAT_SURUMU = 1
KISI_RE = re.compile(r"^[a-z][a-z0-9_]{0,19}$")          # kısa ad; e-posta/boşluk/@ geçersiz
HAT_RE = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
ITER_RE = re.compile(r"^iter_(\d{4})_step_(\d{6})_([a-z][a-z0-9_]{0,19})\.tar$")
SABIT_RE = re.compile(r"^sabit_v(\d+)\.tar$")

ITER_ALLOWED_EXT = {".safetensors", ".json", ".jsonl", ".log"}
SABIT_ALLOWED_EXT = {".safetensors", ".json", ".jsonl", ".log", ".parquet", ".mp4"}
FORBIDDEN_EXT = {".pt", ".pth", ".bin", ".pkl", ".pickle", ".ckpt", ".npy", ".npz", ".so", ".py", ".sh"}
ITER_TOP_FILES = {"manifest.json", "train.log", "attempts.jsonl"}
ITER_DIRS = ("pretrained_model", "training_state")
SABIT_DIRS = ("ham", "act")

CODE_PATHS = ("cloud", "configs/training", "src/mahut_kol/commands")
CREDIT_USD = 30.0
CREDIT_MARGIN_USD = 5.0


class DevirHata(Exception):
    """Kullanıcıya Türkçe gösterilecek, eğitimi durduran hata."""


class CatallanmaHatasi(DevirHata):
    pass


# ----------------------------------------------------------------------------------------------
# Adlar
# ----------------------------------------------------------------------------------------------


def volume_names(env=None) -> tuple[str, str]:
    """(veri Volume'u, çıktı Volume'u). MAHUT_VOLUME_SUFFIX yalnızca teste özeldir."""
    env = os.environ if env is None else env
    suf = env.get("MAHUT_VOLUME_SUFFIX", "")
    return f"mahut-datasets{suf}", f"mahut-runs{suf}"


def check_kisi(kisi: str) -> str:
    if not isinstance(kisi, str) or not KISI_RE.match(kisi):
        raise DevirHata(f"--kisi '{kisi}' geçersiz: küçük harf/rakam/_ ile başlayan en çok 20 karakterlik kısa ad olmalı "
                        "(e-posta, boşluk veya özel karakter yok).")
    return kisi


def check_hat(hat: str) -> str:
    if not isinstance(hat, str) or not HAT_RE.match(hat):
        raise DevirHata(f"--hat '{hat}' geçersiz: küçük harf/rakam/_ ile başlayan en çok 32 karakter (ör. act_v1).")
    return hat


def iter_name(iterasyon: int, adim: int, kisi: str) -> str:
    check_kisi(kisi)
    if not (1 <= iterasyon <= 9999) or not (0 <= adim <= 999999):
        raise DevirHata(f"iterasyon/adım aralık dışı: {iterasyon}/{adim}")
    return f"iter_{iterasyon:04d}_step_{adim:06d}_{kisi}.tar"


def parse_iter_name(name: str) -> dict:
    m = ITER_RE.match(name)
    if not m:
        raise DevirHata(f"'{name}' beklenen ad desenine uymuyor (iter_NNNN_step_NNNNNN_kisi.tar). "
                        "Drive'ın eklediği ' (1)' gibi ekleri silin.")
    return {"iterasyon": int(m[1]), "adim": int(m[2]), "kisi": m[3]}


def sabit_name(surum: int) -> str:
    if surum < 1:
        raise DevirHata("sabit sürümü >= 1 olmalı")
    return f"sabit_v{surum}.tar"


def dataset_names(veri_basename: str, surum: int, has_act: bool) -> dict:
    """Tüm hesaplarda AYNI olan Volume veri adları (checkpoint içindeki mutlak yollar buna bağlı)."""
    ham = f"{veri_basename}_s{surum}"
    return {"ham": ham, **({"act": f"{veri_basename}_act_s{surum}"} if has_act else {})}


# ----------------------------------------------------------------------------------------------
# Özet ve tar
# ----------------------------------------------------------------------------------------------


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                return h.hexdigest()
            h.update(b)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def manifest_digest(manifest: dict) -> str:
    """Manifest'in kanonik özeti (sabit kimliği / geri yükleme işareti için)."""
    return sha256_bytes(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode())


def _check_member_name(name: str, kind: str) -> PurePosixPath:
    if not name or name.startswith("/") or "//" in name or name.endswith("/") or "\\" in name or re.match(r"^[A-Za-z]:", name):
        raise DevirHata(f"tar üyesi güvensiz yol: '{name}' (mutlak yol)")
    p = PurePosixPath(name)
    if ".." in p.parts or "." in p.parts[:-1] or any(part == "" for part in p.parts):
        raise DevirHata(f"tar üyesi güvensiz yol: '{name}' ('..' veya geçersiz bileşen)")
    ext = p.suffix.lower()
    if ext in FORBIDDEN_EXT:
        raise DevirHata(f"tar üyesi yasaklı dosya türü (pickle/çalıştırılabilir olabilir): '{name}'")
    if kind == "iter":
        allowed_ext, tops, dirs = ITER_ALLOWED_EXT, ITER_TOP_FILES, ITER_DIRS
        ok = (len(p.parts) == 1 and name in tops) or (len(p.parts) == 2 and p.parts[0] in dirs)
    elif kind == "sabit":
        allowed_ext, dirs = SABIT_ALLOWED_EXT, SABIT_DIRS
        ok = name == "manifest.json" or (len(p.parts) >= 2 and p.parts[0] in dirs)
    else:
        raise ValueError(kind)
    if not ok:
        raise DevirHata(f"tar üyesi beklenmeyen konumda: '{name}'")
    if ext not in allowed_ext:
        raise DevirHata(f"tar üyesi beklenmeyen dosya türü '{ext}': '{name}'")
    return p


def inspect_tar(tar_path: Path, kind: str) -> dict:
    """Tar'ı AÇMADAN üyelerini denetler; {ad: boyut} döner. Bozuk/yarım tar DevirHata verir."""
    members = {}
    try:
        with tarfile.open(tar_path, "r:") as tf:  # ':' = sıkıştırmasız
            for m in tf:
                if not m.isreg():
                    raise DevirHata(f"tar üyesi düz dosya değil (sembolik/sert bağ, klasör veya aygıt): '{m.name}'")
                _check_member_name(m.name, kind)
                if m.name in members:
                    raise DevirHata(f"tar içinde yinelenen üye: '{m.name}'")
                members[m.name] = m.size
    except (tarfile.TarError, EOFError, OSError) as e:
        raise DevirHata(f"tar okunamıyor (yarım/bozuk indirme olabilir): {tar_path.name}: {e}") from e
    if "manifest.json" not in members:
        raise DevirHata(f"{tar_path.name} içinde manifest.json yok")
    return members


def read_manifest(tar_path: Path, kind: str) -> dict:
    inspect_tar(tar_path, kind)
    with tarfile.open(tar_path, "r:") as tf:
        data = tf.extractfile("manifest.json").read()
    try:
        m = json.loads(data)
    except ValueError as e:
        raise DevirHata(f"manifest.json okunamadı: {e}") from e
    if m.get("format_surumu") != FORMAT_SURUMU or m.get("tur") != kind:
        raise DevirHata(f"manifest biçimi/türü uyumsuz (format={m.get('format_surumu')}, tur={m.get('tur')}); "
                        "aynı git commit'inden çalıştığınızdan emin olun.")
    return m


def verify_tar(tar_path: Path, kind: str) -> dict:
    """Üye denetimi + manifest'teki HER sha256'nın akış üzerinden doğrulanması. Manifest'i döner."""
    members = inspect_tar(tar_path, kind)
    manifest = read_manifest(tar_path, kind)
    files = manifest.get("dosyalar")
    if not isinstance(files, dict) or not files:
        raise DevirHata("manifest'te dosya özetleri (dosyalar) yok")
    listed, actual = set(files), set(members) - {"manifest.json"}
    if listed != actual:
        eksik, fazla = sorted(listed - actual), sorted(actual - listed)
        raise DevirHata(f"tar içeriği manifest ile uyuşmuyor (eksik: {eksik[:3]}, manifest'te olmayan: {fazla[:3]})")
    with tarfile.open(tar_path, "r:") as tf:
        for name, want in sorted(files.items()):
            h = hashlib.sha256()
            fh = tf.extractfile(name)
            size = 0
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                h.update(b)
                size += len(b)
            if h.hexdigest() != want:
                raise DevirHata(f"özet (sha256) tutmuyor: {name} (yarım ya da bozuk indirme). Dosyayı yeniden indirin.")
    return manifest


def extract_tar(tar_path: Path, dest: Path, kind: str, skip=("manifest.json",)) -> list[str]:
    """Doğrulanmış tar'ı `dest` içine çıkarır (elle; hedef dışına yazamaz). Çıkarılan göreli yolları döner."""
    inspect_tar(tar_path, kind)
    dest = Path(dest).resolve()
    out = []
    with tarfile.open(tar_path, "r:") as tf:
        for m in tf:
            if m.name in skip:
                continue
            target = (dest / m.name).resolve()
            if dest not in target.parents:
                raise DevirHata(f"tar üyesi hedef dışına çıkıyor: '{m.name}'")
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "xb") as out_fh:  # 'x': var olan dosyanın üzerine yazma
                src = tf.extractfile(m)
                while True:
                    b = src.read(1 << 20)
                    if not b:
                        break
                    out_fh.write(b)
            out.append(m.name)
    return out


def _write_tar(out_path: Path, files: dict[str, Path], manifest: dict, kind: str) -> None:
    """files: {tar_içi_ad: yerel_yol}. Dosyalar tek tek eklenir (klasör girdisi yok), sıkıştırmasız."""
    out_path = Path(out_path)
    if out_path.exists():
        raise DevirHata(f"{out_path} zaten var; üzerine yazılmaz.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".part")
    if tmp.exists():
        raise DevirHata(f"{tmp} yarım bir yazımdan kalmış; kontrol edip elle taşıyın.")
    mj = json.dumps(manifest, indent=1, sort_keys=True, ensure_ascii=False).encode()
    try:
        with tarfile.open(tmp, "w:", format=tarfile.PAX_FORMAT) as tf:
            info = tarfile.TarInfo("manifest.json")
            info.size, info.mtime, info.mode = len(mj), int(time.time()), 0o644
            import io

            tf.addfile(info, io.BytesIO(mj))
            for arc, p in sorted(files.items()):
                tf.inodes = {}  # aynı inode'u paylaşan (sert bağlı) dosyalar sert bağ OLARAK değil, düz dosya yazılsın
                ti = tf.gettarinfo(str(p), arcname=arc)
                ti.uid = ti.gid = 0
                ti.uname = ti.gname = ""
                ti.mode = 0o644
                with open(p, "rb") as fh:
                    tf.addfile(ti, fh)
        verify_tar(tmp, kind)  # yazılanı doğrula; geçerliyse son adla görünür (yarım/bozuk dosya tar adıyla durmaz)
        os.replace(tmp, out_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def build_iter_tar(out_path: Path, ckpt_dir: Path, extra_files: dict[str, Path], fields: dict) -> dict:
    """pretrained_model/ + training_state/ + (train.log, attempts.jsonl) -> iter tar. Manifest'i döner."""
    files = {}
    for sub in ITER_DIRS:
        for p in sorted((Path(ckpt_dir) / sub).iterdir()):
            if p.is_file():
                files[f"{sub}/{p.name}"] = p
    for arc, p in extra_files.items():
        if arc not in ITER_TOP_FILES - {"manifest.json"}:
            raise DevirHata(f"beklenmeyen ek dosya: {arc}")
        files[arc] = Path(p)
    manifest = {
        "format_surumu": FORMAT_SURUMU, "tur": "iter", **fields,
        "olusturma": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dosyalar": {arc: sha256_file(p) for arc, p in sorted(files.items())},
    }
    _write_tar(Path(out_path), files, manifest, "iter")
    return manifest


def build_sabit_tar(out_path: Path, surum: int, ham: Path, act: Path | None, ad_ham: str, ad_act: str | None) -> dict:
    """Eğitim veri kümesi (+ ACT türevi) -> sabit tar. ACT videoları ham ile özdeşse yalnızca BİR kopya tutulur."""
    keep = ("meta", "data", "videos")
    top = ("mahut_episodes.jsonl", "DERIVED_FROM.json")

    def collect(root: Path, prefix: str):
        d = {}
        for sub in keep:
            for p in sorted((root / sub).rglob("*")) if (root / sub).exists() else []:
                if p.is_file():
                    d[f"{prefix}/{p.relative_to(root).as_posix()}"] = p
        for t in top:
            if (root / t).is_file():
                d[f"{prefix}/{t}"] = root / t
        return d

    files = collect(Path(ham), "ham")
    if not any(k.startswith("ham/meta/info.json") for k in files):
        raise DevirHata(f"{ham} LeRobot veri kümesi değil (meta/info.json yok)")
    sha = {k: sha256_file(p) for k, p in files.items()}
    dedup = {}
    if act is not None:
        act_files = collect(Path(act), "act")
        for k, p in act_files.items():
            h = sha256_file(p)
            twin = "ham/" + k[len("act/"):]
            if k.startswith("act/videos/") and sha.get(twin) == h:
                dedup[k] = twin           # tar'a girmez, kur sırasında ham'dan alınır
                continue
            files[k] = p
            sha[k] = h
    info = json.loads((Path(ham) / "meta" / "info.json").read_text())
    manifest = {
        "format_surumu": FORMAT_SURUMU, "tur": "sabit", "surum": int(surum),
        "veri_kumeleri": {"ham": ad_ham, **({"act": ad_act} if ad_act else {})},
        "act_video_ham_ile_ortak": dedup,
        "bolum_sayisi": int(info.get("total_episodes", 0)), "kare_sayisi": int(info.get("total_frames", 0)),
        "olusturma": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "dosyalar": {k: sha[k] for k in sorted(files)},
    }
    _write_tar(Path(out_path), files, manifest, "sabit")
    return manifest


# ----------------------------------------------------------------------------------------------
# İterasyon zinciri ve çatallanma
# ----------------------------------------------------------------------------------------------


def scan_iterations(hat_dir: Path) -> list[dict]:
    """Yerel hat klasöründeki iter_*.tar: [{yol, ad, iterasyon, adim, kisi, parent_iterasyon, manifest}] (iterasyona göre)."""
    out = []
    hat_dir = Path(hat_dir)
    for p in sorted(hat_dir.glob("iter_*.tar")) if hat_dir.exists() else []:
        meta = parse_iter_name(p.name)
        m = read_manifest(p, "iter")
        if (m.get("iterasyon"), m.get("adim"), m.get("kisi")) != (meta["iterasyon"], meta["adim"], meta["kisi"]):
            raise DevirHata(f"{p.name}: dosya adı manifest ile uyuşmuyor (iterasyon/adım/kişi). Yeniden adlandırılmış olabilir.")
        out.append({"yol": p, "ad": p.name, **meta, "parent_iterasyon": m.get("parent_iterasyon"), "manifest": m})
    return sorted(out, key=lambda r: (r["iterasyon"], r["ad"]))


def pick_parent(hat_dir: Path, sec: str | None = None) -> dict | None:
    """Devam edilecek iterasyon. Çatallanma varsa `sec` olmadan CatallanmaHatasi."""
    its = scan_iterations(hat_dir)
    if sec is not None:
        for r in its:
            if r["ad"] == Path(sec).name:
                return r
        raise DevirHata(f"--sec '{sec}' bu hatta bulunamadı ({Path(hat_dir)}).")
    if not its:
        return None
    problems = []
    nums = [r["iterasyon"] for r in its]
    for n in sorted(set(nums)):
        if nums.count(n) > 1:
            problems.append(f"aynı iterasyon numarası {n}: " + ", ".join(r["ad"] for r in its if r["iterasyon"] == n))
    parents = [r["parent_iterasyon"] for r in its if r["parent_iterasyon"] is not None]
    for pn in sorted(set(parents)):
        if parents.count(pn) > 1:
            problems.append(f"aynı ebeveyni (iterasyon {pn}) olan dosyalar: "
                            + ", ".join(r["ad"] for r in its if r["parent_iterasyon"] == pn))
    if problems:
        raise CatallanmaHatasi("ÇATALLANMA var: " + "; ".join(problems) + ". Hangisinden devam edeceğinizi `--sec <dosya>` ile seçin "
                               "(diğerleri etkilenmez, silinmez).")
    return its[-1]


def next_iteration_number(hat_dir: Path) -> int:
    its = scan_iterations(hat_dir)
    return (max(r["iterasyon"] for r in its) + 1) if its else 1


# ----------------------------------------------------------------------------------------------
# Git denetimi
# ----------------------------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)


def git_head(repo: Path) -> str:
    r = _git(repo, "rev-parse", "HEAD")
    if r.returncode != 0:
        raise DevirHata("git deposu okunamadı (`git clone` ile alınmış bir klasörde çalıştırın).")
    return r.stdout.strip()


def git_dirty_files(repo: Path, exclude=("Data_Model/",)) -> list[str]:
    r = _git(repo, "status", "--porcelain", "--untracked-files=all")
    if r.returncode != 0:
        raise DevirHata("git status çalışmadı.")
    out = []
    for line in r.stdout.splitlines():
        path = line[3:].strip().strip('"')
        if not any(path.startswith(x) for x in exclude):
            out.append(line.strip())
    return out


def check_git(repo: Path, manifest_commit: str | None, paths=CODE_PATHS) -> None:
    dirty = git_dirty_files(repo)
    if dirty:
        raise DevirHata("git çalışma ağacı temiz değil (Data_Model/ hariç):\n  " + "\n  ".join(dirty[:8])
                        + "\nDeğişiklikleri geri alın ya da commit'leyin; herkes aynı kodla çalışmalı.")
    if manifest_commit:
        if _git(repo, "cat-file", "-e", f"{manifest_commit}^{{commit}}").returncode != 0:
            raise DevirHata(f"manifest'teki commit ({manifest_commit[:10]}) bu depoda yok. `git fetch` yapın / doğru dalı alın.")
        for p in paths:
            if _git(repo, "diff", "--quiet", manifest_commit, "HEAD", "--", p).returncode != 0:
                raise DevirHata(f"'{p}' manifest'teki commit ({manifest_commit[:10]}) ile aynı değil. "
                                f"`git checkout {manifest_commit[:10]}` ile aynı koda geçin.")


# ----------------------------------------------------------------------------------------------
# Sürüm, kredi ve maliyet
# ----------------------------------------------------------------------------------------------


def compare_versions(manifest_versions: dict, current: dict) -> list[str]:
    """Farklı paket sürümleri (boş = aynı)."""
    diffs = []
    for k, v in sorted(manifest_versions.items()):
        if current.get(k) != v:
            diffs.append(f"{k}: manifest={v} şimdiki={current.get(k)}")
    return diffs


def check_hat_and_sabit(manifest: dict, hat: str, sabit_manifest_digest: str | None, sabit_surum: int | None) -> None:
    if manifest.get("hat") != hat:
        raise DevirHata(f"hat uyuşmuyor: tar '{manifest.get('hat')}', istenen '{hat}'")
    if sabit_surum is not None and manifest.get("sabit_surumu") != sabit_surum:
        raise DevirHata(f"sabit sürümü uyuşmuyor: tar sabit_v{manifest.get('sabit_surumu')}, "
                        f"elinizdeki sabit_v{sabit_surum}")
    if sabit_manifest_digest and manifest.get("sabit_manifest_ozeti") not in (None, sabit_manifest_digest):
        raise DevirHata("sabit tar içeriği (manifest özeti) iterasyonun üretildiği sabit ile aynı değil.")


def credit_used(summary: dict) -> float:
    """`modal billing summary --json` çıktısından kredi ile karşılanan + faturalanan tutar (USD)."""
    adj = summary.get("adjustments", {})
    return abs(float(adj.get("credits", 0) or 0)) + float(summary.get("billed_cost", 0) or 0)


def available_budget(spent_usd: float, limit: float = CREDIT_USD, margin: float = CREDIT_MARGIN_USD) -> float:
    return max(0.0, limit - margin - spent_usd)


def cost_model(cfg: dict) -> dict:
    c = (cfg.get("maliyet") or {})
    for k in ("usd_per_1000_steps", "sabit_usd", "steps_per_s", "startup_min"):
        if k not in c:
            raise DevirHata(f"configs/training/*.yaml içinde maliyet.{k} yok")
    return {k: float(c[k]) for k in ("usd_per_1000_steps", "sabit_usd", "steps_per_s", "startup_min")}


def estimate_cost(cfg: dict, steps: int) -> float:
    c = cost_model(cfg)
    return c["sabit_usd"] + c["usd_per_1000_steps"] * steps / 1000.0


def paketle_reserve_usd() -> float:
    """`paketle` (indirme) ve imaj/hash çağrıları için ayrılan pay."""
    return 0.10


def max_steps(cfg: dict, available_usd: float) -> int:
    c = cost_model(cfg)
    usable = available_usd - c["sabit_usd"] - paketle_reserve_usd()
    return max(0, int(math.floor(usable / c["usd_per_1000_steps"] * 1000 + 1e-6)))


def timeout_minutes(cfg: dict, steps: int) -> int:
    """Tek deneme zaman aşımı: açılış + adım/hız, %30 pay; 1 dk'dan az olmaz."""
    c = cost_model(cfg)
    return max(1, int(math.ceil(c["startup_min"] + steps / c["steps_per_s"] / 60.0 * 1.3)))


def check_steps_allowed(cfg: dict, steps: int, spent_usd: float, max_timeout_min: int,
                        limit: float = CREDIT_USD, margin: float = CREDIT_MARGIN_USD) -> int:
    """--adim denetimi; izin verilirse timeout (dk) döner."""
    if steps < 1:
        raise DevirHata("--adim en az 1 olmalı")
    avail = available_budget(spent_usd, limit, margin)
    mx = max_steps(cfg, avail)
    if steps > mx:
        raise DevirHata(f"--adim {steps} kredi sınırını aşıyor: bu ay harcanan {spent_usd:.2f} USD, ayrılan güvenlik payı "
                        f"{margin:.0f} USD (aylık {limit:.0f} USD). İzin verilen azami adım: {mx}.")
    t = timeout_minutes(cfg, steps)
    if t > max_timeout_min:
        raise DevirHata(f"--adim {steps} için gereken zaman aşımı {t} dk > --max-timeout-min={max_timeout_min}. "
                        f"Adımı azaltın (≈{int((max_timeout_min - cost_model(cfg)['startup_min']) * 60 * cost_model(cfg)['steps_per_s'] / 1.3)}) "
                        "ya da sınırı bilerek yükseltin.")
    return t
