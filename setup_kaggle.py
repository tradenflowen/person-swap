"""
setup_kaggle.py  —  Person-Swap Pipeline: Kaggle Environment Bootstrap
=======================================================================
Run this as the FIRST cell in every new Kaggle session.
"""

import os
import sys
import subprocess
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR   = "/kaggle/working"
REPO_DIR   = f"{WORK_DIR}/person-swap"
CKPT_DIR   = f"{REPO_DIR}/checkpoints"
EXT_DIR    = f"{WORK_DIR}/external"

PERSON_SWAP_REPO = "https://github.com/tradenflowen/person-swap.git"

# ── External GitHub repos ─────────────────────────────────────────────────────
EXTERNAL_REPOS = [
    ("https://github.com/facebookresearch/segment-anything-2.git", "sam2", ""),
    ("https://github.com/AIRI-Institute/HairFastGAN.git", "HairFastGAN", ""),
    ("https://github.com/yangxy/GPEN.git", "GPEN", ""),
    ("https://github.com/levihsu/OOTDiffusion.git", "OOTDiffusion", ""),
    ("https://github.com/princeton-vl/RAFT.git", "RAFT", "core"),
    ("https://github.com/sczhou/ProPainter.git", "ProPainter", ""),
    ("https://github.com/hzwer/ECCV2022-RIFE.git", "RIFE", ""),
]

# ── InstantID files (wget) ────────────────────────────────────────────────────
INSTANTID_FILES = [
    ("https://raw.githubusercontent.com/instantX-research/InstantID/main/pipeline_stable_diffusion_xl_instantid.py", "pipeline_stable_diffusion_xl_instantid.py"),
    ("https://raw.githubusercontent.com/instantX-research/InstantID/main/ip_adapter/__init__.py", "ip_adapter/__init__.py"),
    ("https://raw.githubusercontent.com/instantX-research/InstantID/main/ip_adapter/attention_processor.py", "ip_adapter/attention_processor.py"),
    ("https://raw.githubusercontent.com/instantX-research/InstantID/main/ip_adapter/ip_adapter.py", "ip_adapter/ip_adapter.py"),
    ("https://raw.githubusercontent.com/instantX-research/InstantID/main/ip_adapter/resampler.py", "ip_adapter/resampler.py"),
]

# ── pip packages ───────────────────────────────────────────────────────────────
PIP_PACKAGES = [
    "hydra-core", "iopath", "mediapipe", "insightface", "onnxruntime-gpu",
    "diffusers>=0.27.0", "transformers>=4.36.0", "accelerate", "safetensors",
    "ninja", "scipy", "scikit-image", "realesrgan", "basicsr",
    "opencv-python-headless", "Pillow", "numpy", "huggingface_hub", "einops", "timm",
]

# ── Model checkpoints ─────────────────────────────────────────────────────────
CHECKPOINTS = [
    (None, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt", "sam2_hiera_small.pt"),
    (None, "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task", "pose_landmarker.task"),
    ("hf", "InstantX/InstantID", "ip-adapter.bin", "instantid"),
    ("hf", "InstantX/InstantID", "ControlNetModel/config.json", "instantid"),
    ("hf", "InstantX/InstantID", "ControlNetModel/diffusion_pytorch_model.safetensors", "instantid"),
    ("hf_snapshot", "AIRI-Institute/HairFastGAN", "hairfastgan"),
    (None, "https://github.com/yangxy/GPEN/releases/download/v1.0/GPEN-BFR-512.pth", "gpen/GPEN-BFR-512.pth"),
    (None, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", "realesrgan/RealESRGAN_x4plus.pth"),
]

STATUS = {}

def run(cmd, desc="", check=True):
    print(f"  → {desc or cmd[:80]}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0 and check:
        print(f"  ✗ FAILED:\n{result.stderr[-1000:]}")
    return result.returncode == 0

def add_path(p):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
        return True
    return False

def step_clone_repo():
    print("\n[1/7] person-swap repo")
    if os.path.isdir(REPO_DIR):
        print("  Already cloned — pulling latest...")
        run(f"git -C {REPO_DIR} pull --ff-only", "git pull")
    else:
        ok = run(f"git clone {PERSON_SWAP_REPO} {REPO_DIR}", "git clone person-swap")
        if not ok:
            STATUS["repo"] = False
            return
    STATUS["repo"] = os.path.isdir(f"{REPO_DIR}/pipeline")
    add_path(REPO_DIR)
    add_path(f"{REPO_DIR}/pipeline")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"  ✓ Repo ready at {REPO_DIR}")

def step_clone_externals():
    print("\n[2/7] External GitHub repos")
    os.makedirs(EXT_DIR, exist_ok=True)
    for clone_url, folder, subpath in EXTERNAL_REPOS:
        dest = f"{EXT_DIR}/{folder}"
        if os.path.isdir(dest):
            print(f"  {folder}: already present")
        else:
            ok = run(f"git clone --depth 1 {clone_url} {dest}", f"clone {folder}")
            if not ok:
                STATUS[f"ext_{folder}"] = False
                continue
        path_to_add = dest if subpath == "" else f"{dest}/{subpath}"
        add_path(path_to_add)
        STATUS[f"ext_{folder}"] = os.path.isdir(path_to_add)
        print(f"  ✓ {folder} → sys.path: {path_to_add}")

def step_fetch_instantid():
    print("\n[3/7] InstantID files (wget)")
    instantid_dir = f"{EXT_DIR}/InstantID"
    os.makedirs(instantid_dir, exist_ok=True)
    all_ok = True
    for raw_url, local_name in INSTANTID_FILES:
        dest = f"{instantid_dir}/{local_name}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isfile(dest):
            print(f"  ✓ {local_name}: already present")
            continue
        ok = run(f"wget -q {raw_url} -O {dest}", f"wget {local_name}")
        if ok:
            print(f"  ✓ {local_name}: downloaded")
        else:
            print(f"  ✗ {local_name}: FAILED")
            all_ok = False
    add_path(instantid_dir)
    STATUS["ext_InstantID"] = all_ok
    print(f"  {'✓' if all_ok else '⚠'} InstantID → sys.path: {instantid_dir}")

def step_pip_install():
    print("\n[4/7] pip packages")
    sam2_dir = f"{EXT_DIR}/sam2"
    if os.path.isdir(sam2_dir):
        run(f"pip install -e {sam2_dir} -q", "pip install sam2 (from clone)")
    pkg_str = " ".join(f'"{p}"' for p in PIP_PACKAGES)
    run(f"pip install {pkg_str} -q", "core pip packages")
    STATUS["pip"] = True
    print("  ✓ pip installs complete")

def step_download_checkpoints():
    print("\n[5/7] Model checkpoints")
    import urllib.request
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("  ✗ huggingface_hub not available")
        return
    
    for entry in CHECKPOINTS:
        kind = entry[0]
        if kind is None:
            _, url, filename = entry
            dest = f"{CKPT_DIR}/{filename}"
            # CREATE PARENT DIR
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            if os.path.isfile(dest):
                print(f"  ✓ {os.path.basename(filename)}: already present")
                continue
            print(f"  Downloading {os.path.basename(filename)}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  ✓ {os.path.basename(filename)}: downloaded")
            except Exception as e:
                print(f"  ✗ {os.path.basename(filename)}: FAILED — {e}")
        elif kind == "hf":
            _, repo_id, filename, subdir = entry
            dest_dir = f"{CKPT_DIR}/{subdir}"
            dest_file = f"{dest_dir}/{os.path.basename(filename)}"
            if os.path.isfile(dest_file):
                print(f"  ✓ {os.path.basename(filename)}: already present")
                continue
            os.makedirs(dest_dir, exist_ok=True)
            print(f"  Downloading {os.path.basename(filename)} from {repo_id}...")
            try:
                hf_hub_download(repo_id=repo_id, filename=filename, local_dir=dest_dir)
                print(f"  ✓ {os.path.basename(filename)}: downloaded")
            except Exception as e:
                print(f"  ✗ {os.path.basename(filename)}: FAILED — {e}")
        elif kind == "hf_snapshot":
            _, repo_id, subdir = entry
            dest_dir = f"{CKPT_DIR}/{subdir}"
            if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                print(f"  ✓ {subdir} snapshot: already present")
                continue
            os.makedirs(dest_dir, exist_ok=True)
            print(f"  Downloading {repo_id} snapshot...")
            try:
                snapshot_download(repo_id=repo_id, local_dir=dest_dir)
                print(f"  ✓ {subdir}: downloaded")
            except Exception as e:
                print(f"  ✗ {subdir}: FAILED — {e}")
    STATUS["checkpoints"] = True
    print("  ✓ Checkpoint step complete")

def step_wire_paths():
    print("\n[6/7] Wiring sys.path")
    paths = [
        REPO_DIR, f"{REPO_DIR}/pipeline", f"{EXT_DIR}/InstantID",
        f"{EXT_DIR}/sam2", f"{EXT_DIR}/HairFastGAN", f"{EXT_DIR}/GPEN",
        f"{EXT_DIR}/OOTDiffusion", f"{EXT_DIR}/RAFT/core",
        f"{EXT_DIR}/ProPainter", f"{EXT_DIR}/RIFE",
        f"{CKPT_DIR}/hairfastgan", f"{CKPT_DIR}/gpen",
    ]
    for p in paths:
        if os.path.isdir(p):
            add_path(p)
            print(f"  ✓ {p}")
        else:
            print(f"  ⚠ Not found: {p}")
    STATUS["paths"] = True

def patch_insightface():
    print("\n[7/7] Patching InsightFace")
    try:
        import insightface
        init_path = os.path.join(os.path.dirname(insightface.__file__), "app", "__init__.py")
        if not os.path.isfile(init_path):
            return
        content = open(init_path).read()
        if "# PATCHED" in content or "mask_renderer" not in content:
            print("  ✓ Already patched or clean")
            return
        patched = content.replace(
            "from .mask_renderer import MaskRenderer",
            "# PATCHED\n# from .mask_renderer import MaskRenderer"
        )
        open(init_path, "w").write(patched)
        print("  ✓ InsightFace mask_renderer patched")
    except Exception as e:
        print(f"  ⚠ Patch skipped: {e}")

def print_report():
    print("\n" + "="*55)
    print("  ENVIRONMENT READINESS REPORT")
    print("="*55)
    checks = [
        ("person-swap repo", STATUS.get("repo", False)),
        ("SAM2 clone", STATUS.get("ext_sam2", False)),
        ("InstantID files", STATUS.get("ext_InstantID", False)),
        ("HairFastGAN clone", STATUS.get("ext_HairFastGAN", False)),
        ("GPEN clone", STATUS.get("ext_GPEN", False)),
        ("pip packages", STATUS.get("pip", False)),
        ("checkpoints", STATUS.get("checkpoints", False)),
        ("sys.path wired", STATUS.get("paths", False)),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  {'✓' if ok else '✗'}  {label}")
        if not ok:
            all_ok = False
    print("="*55)
    print(f"  {'✅ All systems ready' if all_ok else '⚠ Some steps failed'}")
    print("="*55 + "\n")

def main():
    t0 = time.time()
    print("Person-Swap Kaggle Bootstrap")
    print(f"Working dir: {WORK_DIR}")
    print(f"Repo dir:    {REPO_DIR}")
    print(f"Checkpoints: {CKPT_DIR}")
    print(f"Externals:   {EXT_DIR}\n")
    
    step_clone_repo()
    step_clone_externals()
    step_fetch_instantid()  # ← WAS MISSING
    step_pip_install()
    step_download_checkpoints()
    step_wire_paths()
    patch_insightface()
    print_report()
    
    print(f"Bootstrap completed in {int(time.time() - t0)}s\n")

if __name__ == "__main__":
    main()
else:
    main()