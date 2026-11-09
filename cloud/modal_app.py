"""
Modal uygulaması: VLA eğitimi (SmolVLA / ACT) için imaj, Volume'lar ve uzak fonksiyonlar.

Doğrudan çalıştırmayın; `cloud/mahut_cloud.py` sarmalayıcısını kullanın (GPU/timeout ortam
değişkeniyle bu dosyaya aktarılır, çünkü Modal dekoratör argümanları import anında okunur).

Volume'lar:
    mahut-datasets  /data   sürümlü LeRobot veri kümeleri (ör. mahut_smoke_v1, mahut_smoke_v1_act)
    mahut-runs      /runs   runs/<koşu_adı>/... (checkpoint, log) ve hf_cache/ (HF model önbelleği)

Güvenlik: bu dosyadaki hiçbir fonksiyon Volume'da dosya silmez/ezmez. Koşu klasörü varsa
(`train_entry.prepare_run`) yalnızca aynı launch_id (otomatik yeniden başlatma) ya da açık
`resume` ile sürdürülür, aksi halde hata verir.

Modal fiyatları (modal.com/pricing, 2026-11-09; saniye başına USD) maliyet tahmini içindir.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

GPU = os.environ.get("MAHUT_GPU", "L4")  # tek ad ya da yedekli liste: "L4,A10" (kapasite yoksa sıradakini dener)
GPU_SPEC = GPU.split(",") if "," in GPU else GPU
TIMEOUT_MIN = int(os.environ.get("MAHUT_TIMEOUT_MIN", "30"))
MAX_RETRIES = int(os.environ.get("MAHUT_MAX_RETRIES", "2"))

# modal.com/pricing (2026-11-09), USD / saniye
GPU_USD_PER_S = {
    "T4": 0.000164, "L4": 0.000222, "A10": 0.000306, "L40S": 0.000542, "A100-40GB": 0.000583,
    "A100-80GB": 0.000694, "RTX-PRO-6000": 0.000842, "H100": 0.001097, "H200": 0.001261,
}
CPU_USD_PER_CORE_S = 0.0000131   # fiziksel çekirdek; konteyner >= 0.125 çekirdek
MEM_USD_PER_GIB_S = 0.00000222

app = modal.App("mahut-vla-train")
data_vol = modal.Volume.from_name("mahut-datasets", create_if_missing=True)
runs_vol = modal.Volume.from_name("mahut-runs", create_if_missing=True)

# Yerel ~/.venvs/lerobot ile karşılaştırma: docs'ta / YAPILANLAR.md'de "Modal imajı" bölümünde.
image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0", "git")
    .pip_install(
        "torch==2.6.0", "torchvision==0.21.0",
        "lerobot[smolvla]==0.4.4",
        "transformers==4.57.6", "accelerate==1.15.0", "safetensors==0.8.0", "huggingface-hub==0.35.3",
        "datasets==4.8.5", "diffusers==0.35.2", "num2words==0.5.14", "av==15.1.0", "numpy==2.2.6",
        "pyarrow==25.0.1", "einops==0.8.2", "draccus==0.10.0", "pillow==12.3.0",
        "opencv-python-headless==4.12.0.88", "wandb==0.24.2", "pyyaml==6.0.3",
    )
    .env({
        "HF_HOME": "/runs/hf_cache",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "WANDB_MODE": "disabled",
    })
    .add_local_file(str(HERE / "train_entry.py"), "/root/train_entry.py")
)

COMMON = dict(
    image=image,
    volumes={"/data": data_vol, "/runs": runs_vol},
)


def _gpu_key(device_name: str) -> str:
    """torch cihaz adı -> fiyat tablosu anahtarı (yedekli GPU listesinde gerçek GPU'ya göre)."""
    n = device_name.upper()
    for k in sorted(GPU_USD_PER_S, key=len, reverse=True):
        if k.replace("-", " ").split()[0] in n.split():
            return k
    return GPU.split(",")[0]


def cost_usd(gpu: str, seconds: float, cpu_cores: float = 0.125, mem_gib: float = 0.5) -> float:
    """Tahmini maliyet (GPU + asgari CPU/bellek). CPU/bellek gerçek kullanıma göre biraz artabilir."""
    return seconds * (GPU_USD_PER_S.get(gpu, 0.0) + cpu_cores * CPU_USD_PER_CORE_S + mem_gib * MEM_USD_PER_GIB_S)


@app.function(timeout=15 * 60, **COMMON)
def env_info() -> dict:
    """Konteyner sürümleri (imaj doğrulaması) - GPU'suz, ucuz."""
    import importlib.metadata as md

    pk = ["torch", "torchvision", "lerobot", "transformers", "numpy", "av", "datasets", "accelerate",
          "safetensors", "huggingface-hub", "diffusers", "torchcodec", "pyarrow", "draccus"]
    out = {p: md.version(p) for p in pk}
    out["python"] = sys.version.split()[0]
    out["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    return out


@app.function(gpu=GPU_SPEC, timeout=TIMEOUT_MIN * 60,
              retries=modal.Retries(max_retries=MAX_RETRIES, initial_delay=5.0, backoff_coefficient=1.0),
              **COMMON)
def train(cfg: dict, run_name: str, launch_id: str, resume: bool) -> dict:
    """Tek eğitim denemesi. Kesilirse (timeout/preemption) Modal aynı girdiyle yeniden başlatır;
    aynı launch_id olduğu için train_entry koşuyu son checkpoint'ten sürdürür."""
    import shutil
    import torch

    t0 = time.time()
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok"
    print(f"GPU: {gpu_name} | modal gpu={GPU} | koşu={run_name} launch={launch_id[:8]}", flush=True)

    # Veri kümesini hızlı yerel diske kopyala (Volume'dan okuma yerine); Volume'a dokunulmaz.
    src = Path("/data") / cfg["dataset"]
    if not (src / "meta" / "info.json").exists():
        raise FileNotFoundError(f"Volume 'mahut-datasets' içinde {cfg['dataset']} yok (önce upload).")
    local_root = Path("/tmp/ds") / cfg["dataset"]
    if not local_root.exists():
        shutil.copytree(src, local_root)
    print(f"veri kopyalandı: {local_root} ({time.time() - t0:.0f} sn)", flush=True)

    stop = threading.Event()

    def commit_loop():  # kalıcılık: checkpoint'ler düzenli aralıkla Volume'a işlenir
        while not stop.wait(45):
            try:
                runs_vol.commit()
            except Exception as e:  # noqa: BLE001
                print("commit uyarısı:", e, flush=True)

    th = threading.Thread(target=commit_loop, daemon=True)
    th.start()
    sys.path.insert(0, "/root")
    import train_entry

    argv = ["--config", "/tmp/cfg.json", "--run", run_name, "--data-root", str(local_root),
            "--out-root", "/runs/runs", "--launch-id", launch_id]
    if resume:
        argv.append("--resume")
    Path("/tmp/cfg.json").write_text(json.dumps(cfg))
    Path("/runs/runs").mkdir(parents=True, exist_ok=True)
    try:
        rc = train_entry.main(argv)
    finally:
        stop.set()
        runs_vol.commit()
    wall = time.time() - t0
    att_file = Path("/runs/runs") / run_name / "attempts.jsonl"
    if not att_file.exists():
        raise RuntimeError(f"eğitim sonuç yazmadan bitti (rc={rc})")
    last = json.loads(att_file.read_text().strip().splitlines()[-1])
    summary = {
        "run": run_name, "returncode": rc, "gpu_modal": GPU, "gpu_device": gpu_name,
        "wall_s": round(wall, 1), "train_wall_s": last["wall_s"],
        "steps_per_s": last.get("steps_per_s_from_log"), "resumed": last["resumed"],
        "last_step": last["last_logged_step"], "est_cost_usd": round(cost_usd(_gpu_key(gpu_name), wall), 3),
        "loss_trace": last["losses"],
    }
    print("ÖZET", json.dumps({k: v for k, v in summary.items() if k != "loss_trace"}), flush=True)
    if rc != 0:
        raise RuntimeError(f"eğitim hata kodu {rc}")
    return summary


@app.local_entrypoint()
def train_cli(config_json: str, run: str, launch_id: str, resume: bool = False):
    """`modal run --detach cloud/modal_app.py::train_cli --config-json ... --run ...`"""
    s = train.remote(json.loads(config_json), run, launch_id, resume)
    print("TAMAM", json.dumps({k: v for k, v in s.items() if k != "loss_trace"}))
    print("kayıp izi:", s["loss_trace"][:3], "...", s["loss_trace"][-3:])


@app.local_entrypoint()
def info():
    print(json.dumps(env_info.remote(), indent=1))
