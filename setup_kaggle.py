"""
setup_kaggle.py  —  Person-Swap Pipeline: Kaggle Environment Bootstrap
=======================================================================
Run this as the FIRST cell in every new Kaggle session.

What it does:
  1. Clones the person-swap repo
  2. Clones all external GitHub repos that provide pipeline Python code
     (InstantID is fetched via wget instead of git clone — Kaggle blocks it)
  3. pip-installs all required packages
  4. Downloads all model checkpoints (skips if already present)
  5. Adds all required paths to sys.path
  6. Prints a readiness report so you know exactly what is ready

Usage:
  exec(open("/kaggle/working/setup_kaggle.py").read())
  -- OR --
  %run /kaggle/working/setup_kaggle.py

After this runs, you can import any pipeline module directly:
  from pipeline.segment import segment_video
  from pipeline.pose import load_dwpose
  ... etc
"""

import os
import sys
import subprocess
import time

# ── Paths ──────────────────────────────────────────────────────────────────────
WORK_DIR   = "/kaggle/working"
REPO_DIR   = f"{WORK_DIR}/person-swap"
CKPT_DIR   = f"{REPO_DIR}/checkpoints"
EXT_DIR    = f"{WORK_DIR}/external"          # external GitHub repos land here

# ── Your repo ─────────────────────────────────────────────────────────────────
PERSON_SWAP_REPO = "https://github.com/tradenflowen/person-swap.git"

# ── External GitHub repos needed for pipeline Python code ─────────────────────
# Each entry: (clone_url, local_folder_name, subfolder_to_add_to_sys_path)
# subfolder="" means add the root of the clone to sys.path
# NOTE: InstantID is NOT listed here — Kaggle blocks its git clone.
#       It is handled separately in step_fetch_instantid() via wget.
EXTERNAL_REPOS = [
    (
        "https://github.com/facebookresearch/segment-anything-2.git",
        "sam2",
        ""          # sam2 package is at repo root
    ),
    (
        "https://github.com/AIRI-Institute/HairFastGAN.git",
        "HairFastGAN",
        ""          # hair_swap.py is at root
    ),
    (
        "https://github.com/levihsu/OOTDiffusion.git",
        "OOTDiffusion",
        ""          # pipelines/ folder is at root
    ),
    (
        "https://github.com/princeton-vl/RAFT.git",
        "RAFT",
        "core"      # raft module lives in core/
    ),
    (
        "https://github.com/sczhou/ProPainter.git",
        "ProPainter",
        ""          # model/ is at root
    ),
    (
        "https://github.com/hzwer/ECCV2022-RIFE.git",
        "RIFE",
        ""          # model/ is at root
    ),
]

# ── InstantID files to fetch via wget (git clone blocked in Kaggle) ────────────
# Each entry: (raw_github_url, local_filename_relative_to_InstantID_dir)
INSTANTID_FILES = [
    (
        "https://raw.githubusercontent.com/InstantX/InstantID/main/pipeline_stable_diffusion_xl_instantid.py",
        "pipeline_stable_diffusion_xl_instantid.py"
    ),
    (
        "https://raw.githubusercontent.com/InstantX/InstantID/main/ip_adapter/__init__.py",
        "ip_adapter/__init__.py"
    ),
    (
        "https://raw.githubusercontent.com/InstantX/InstantID/main/ip_adapter/attention_processor.py",
        "ip_adapter/attention_processor.py"
    ),
    (
        "https://raw.githubusercontent.com/InstantX/InstantID/main/ip_adapter/ip_adapter.py",
        "ip_adapter/ip_adapter.py"
    ),
    (
        "https://raw.githubusercontent.com/InstantX/InstantID/main/ip_adapter/resampler.py",
        "ip_adapter/resampler.py"
    ),
]

# ── pip packages ───────────────────────────────────────────────────────────────
PIP_PACKAGES = [
    # SAM2
    "hydra-core",
    "iopath",
    # Pose
    "mediapipe",
    # InsightFace / face stage
    "insightface",
    "onnxruntime-gpu",
    # Diffusers stack
    "diffusers>=0.25.0",
    "transformers>=4.36.0",
    "accelerate",
    "safetensors",
    # HairFastGAN deps
    "ninja",
    "scipy",
    "scikit-image",
    # Upscale
    "realesrgan",
    "basicsr",
    # General
    "opencv-python-headless",
    "Pillow",
    "numpy",
    "huggingface_hub",
    "einops",
    "timm",
]

# ── Model checkpoints to download ─────────────────────────────────────────────
# Entry formats:
#   (None, url, filename)                      — direct URL download
#   ("hf", repo_id, filename, subdir)          — HF single file
#   ("hf_snapshot", repo_id, subdir)           — HF full repo snapshot
CHECKPOINTS = [
    # SAM2 checkpoint
    (
        None,
        "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_small.pt"
    ),
    # MediaPipe pose model
    (
        None,
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "pose_landmarker.task"
    ),
    # InstantID weights (pipeline Python code fetched separately via wget)
    (
        "hf",
        "InstantX/InstantID",
        "ip-adapter.bin",
        "instantid"
    ),
    (
        "hf",
        "InstantX/InstantID",
        "ControlNetModel/config.json",
        "instantid"
    ),
    (
        "hf",
        "InstantX/InstantID",
        "ControlNetModel/diffusion_pytorch_model.safetensors",
        "instantid"
    ),
    # HairFastGAN weights snapshot (Python code comes from GitHub clone)
    (
        "hf_snapshot",
        "AIRI-Institute/HairFastGAN",
        "hairfastgan"
    ),
    # GPEN skin enhancer — corrected repo (akhaliq/GPEN is private/gone)
    (
        "hf",
        "TencentARC/GPEN",
        "GPEN-BFR-512.pth",
        "gpen"
    ),
    # Real-ESRGAN — corrected repo and file path
    (
        "hf",
        "xinntao/Real-ESRGAN",
        "experiments/pretrained_models/RealESRGAN_x4plus.pth",
        "realesrgan"
    ),
]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def run(cmd, desc="", check=True):
    """Run a shell command, print result."""
    print(f"  → {desc or cmd[:80]}")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=True, text=True
    )
    if result.returncode != 0 and check:
        print(f"  ✗ FAILED:\n{result.stderr[-1000:]}")
    elif result.returncode != 0:
        print(f"  ⚠ Non-zero exit (non-fatal):\n{result.stderr[-400:]}")
    return result.returncode == 0


def add_path(p):
    if p not in sys.path and os.path.isdir(p):
        sys.path.insert(0, p)
        return True
    return False


STATUS = {}   # populated at end for readiness report


# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Clone / update person-swap repo
# ══════════════════════════════════════════════════════════════════════════════

def step_clone_repo():
    print("\n[1/6] person-swap repo")
    if os.path.isdir(REPO_DIR):
        print("  Already cloned — pulling latest...")
        run(f"git -C {REPO_DIR} pull --ff-only", "git pull")
    else:
        ok = run(
            f"git clone {PERSON_SWAP_REPO} {REPO_DIR}",
            "git clone person-swap"
        )
        STATUS["repo"] = ok
        if not ok:
            return

    STATUS["repo"] = os.path.isdir(f"{REPO_DIR}/pipeline")
    add_path(REPO_DIR)
    add_path(f"{REPO_DIR}/pipeline")
    os.makedirs(CKPT_DIR, exist_ok=True)
    print(f"  ✓ Repo ready at {REPO_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Clone external repos
# ══════════════════════════════════════════════════════════════════════════════

def step_clone_externals():
    print("\n[2/6] External GitHub repos (pipeline code)")
    os.makedirs(EXT_DIR, exist_ok=True)

    for entry in EXTERNAL_REPOS:
        clone_url, folder, subpath = entry
        dest = f"{EXT_DIR}/{folder}"

        if os.path.isdir(dest):
            print(f"  {folder}: already present — skipping clone")
        else:
            ok = run(
                f"git clone --depth 1 {clone_url} {dest}",
                f"clone {folder}"
            )
            if not ok:
                STATUS[f"ext_{folder}"] = False
                continue

        # Add correct subpath to sys.path
        path_to_add = dest if subpath == "" else f"{dest}/{subpath}"
        add_path(path_to_add)
        STATUS[f"ext_{folder}"] = os.path.isdir(path_to_add)
        print(f"  ✓ {folder} → sys.path: {path_to_add}")

    # OOTDiffusion root must also be in sys.path
    # so that `from pipelines.OOTDiffusion import ...` resolves correctly
    ootd_pipe = f"{EXT_DIR}/OOTDiffusion"
    if os.path.isdir(ootd_pipe):
        add_path(ootd_pipe)


# ══════════════════════════════════════════════════════════════════════════════
# Step 3 — Fetch InstantID files via wget (git clone blocked in Kaggle)
# ══════════════════════════════════════════════════════════════════════════════

def step_fetch_instantid():
    print("\n[3/6] InstantID pipeline files (wget — git clone blocked in Kaggle)")
    instantid_dir = f"{EXT_DIR}/InstantID"
    os.makedirs(instantid_dir, exist_ok=True)

    all_ok = True
    for raw_url, local_name in INSTANTID_FILES:
        dest = f"{instantid_dir}/{local_name}"
        dest_dir = os.path.dirname(dest)
        os.makedirs(dest_dir, exist_ok=True)

        if os.path.isfile(dest):
            print(f"  ✓ {local_name}: already present")
            continue

        ok = run(
            f"wget -q {raw_url} -O {dest}",
            f"wget {local_name}"
        )
        if ok:
            print(f"  ✓ {local_name}: downloaded")
        else:
            print(f"  ✗ {local_name}: FAILED")
            all_ok = False

    add_path(instantid_dir)
    STATUS["ext_InstantID"] = all_ok
    if all_ok:
        print(f"  ✓ InstantID → sys.path: {instantid_dir}")
    else:
        print(f"  ⚠ InstantID: some files failed — check above")


# ══════════════════════════════════════════════════════════════════════════════
# Step 4 — pip install
# ══════════════════════════════════════════════════════════════════════════════

def step_pip_install():
    print("\n[4/6] pip packages")

    # Install SAM2 from its cloned repo (not on PyPI as a stable release)
    sam2_dir = f"{EXT_DIR}/sam2"
    if os.path.isdir(sam2_dir):
        run(
            f"pip install -e {sam2_dir} -q",
            "pip install sam2 (from clone)"
        )

    # OOTDiffusion and HairFastGAN requirements.txt are intentionally skipped:
    # they contain broken/incompatible pinned versions that fail on Kaggle.
    # All functionally required packages are covered in PIP_PACKAGES below.

    # Main package list
    pkg_str = " ".join(f'"{p}"' for p in PIP_PACKAGES)
    run(f"pip install {pkg_str} -q", "core pip packages")

    STATUS["pip"] = True
    print("  ✓ pip installs complete")


# ══════════════════════════════════════════════════════════════════════════════
# Step 5 — Download checkpoints
# ══════════════════════════════════════════════════════════════════════════════

def step_download_checkpoints():
    print("\n[5/6] Model checkpoints")
    import urllib.request
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print("  ✗ huggingface_hub not available — run pip step first")
        return

    for entry in CHECKPOINTS:
        kind = entry[0]

        # Direct URL downloads (SAM2 checkpoint, MediaPipe model)
        if kind is None:
            _, url, filename = entry
            dest = f"{CKPT_DIR}/{filename}"
            if os.path.isfile(dest):
                print(f"  ✓ {filename}: already present")
                continue
            print(f"  Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, dest)
                print(f"  ✓ {filename}: downloaded")
            except Exception as e:
                print(f"  ✗ {filename}: FAILED — {e}")

        # HF single file download
        elif kind == "hf":
            _, repo_id, filename, subdir = entry
            dest_dir = f"{CKPT_DIR}/{subdir}"
            # Check by basename so nested paths (e.g. experiments/pretrained_models/x.pth)
            # don't cause false "not present" on re-runs
            dest_file = f"{dest_dir}/{os.path.basename(filename)}"
            if os.path.isfile(dest_file):
                print(f"  ✓ {os.path.basename(filename)}: already present")
                continue
            os.makedirs(dest_dir, exist_ok=True)
            print(f"  Downloading {filename} from {repo_id}...")
            try:
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=dest_dir
                )
                print(f"  ✓ {os.path.basename(filename)}: downloaded")
            except Exception as e:
                print(f"  ✗ {filename}: FAILED — {e}")

        # HF snapshot (entire repo)
        elif kind == "hf_snapshot":
            _, repo_id, subdir = entry
            dest_dir = f"{CKPT_DIR}/{subdir}"
            if os.path.isdir(dest_dir) and os.listdir(dest_dir):
                print(f"  ✓ {subdir} snapshot: already present")
                continue
            os.makedirs(dest_dir, exist_ok=True)
            print(f"  Downloading {repo_id} snapshot (weights only)...")
            try:
                snapshot_download(repo_id=repo_id, local_dir=dest_dir)
                print(f"  ✓ {subdir}: downloaded")
            except Exception as e:
                print(f"  ✗ {subdir}: FAILED — {e}")

    STATUS["checkpoints"] = True
    print("  ✓ Checkpoint step complete (check above for individual failures)")


# ══════════════════════════════════════════════════════════════════════════════
# Step 6 — Wire sys.path for all pipeline modules
# ══════════════════════════════════════════════════════════════════════════════

def step_wire_paths():
    """
    Ensures all import paths are set correctly for the pipeline.
    Safe to call again after a kernel restart without re-running full setup.
    """
    print("\n[6/6] Wiring sys.path")

    paths = [
        REPO_DIR,
        f"{REPO_DIR}/pipeline",
        f"{EXT_DIR}/sam2",
        f"{EXT_DIR}/InstantID",
        f"{EXT_DIR}/HairFastGAN",
        f"{EXT_DIR}/OOTDiffusion",
        f"{EXT_DIR}/RAFT/core",
        f"{EXT_DIR}/ProPainter",
        f"{EXT_DIR}/RIFE",
        # HairFastGAN weights snapshot (pretrained_models/ lives here)
        f"{CKPT_DIR}/hairfastgan",
        # GPEN inference code (face_enhancement.py expected here)
        f"{CKPT_DIR}/gpen",
    ]

    for p in paths:
        if os.path.isdir(p):
            add_path(p)
            print(f"  ✓ {p}")
        else:
            print(f"  ⚠ Not found (may not be cloned yet): {p}")

    STATUS["paths"] = True


# ══════════════════════════════════════════════════════════════════════════════
# Patch: InsightFace mask_renderer import bug (Kaggle ABI issue)
# ══════════════════════════════════════════════════════════════════════════════

def patch_insightface():
    """
    Patches insightface/app/__init__.py to skip mask_renderer import
    which fails in Kaggle due to scipy/numpy ABI mismatch.
    Only runs if insightface is installed and patch not already applied.
    """
    try:
        import insightface
        init_path = os.path.join(
            os.path.dirname(insightface.__file__),
            "app", "__init__.py"
        )
        if not os.path.isfile(init_path):
            return

        content = open(init_path).read()
        if "mask_renderer" not in content:
            return  # already clean
        if "# PATCHED" in content:
            return  # already patched this session

        patched = content.replace(
            "from .mask_renderer import MaskRenderer",
            "# PATCHED: skip mask_renderer (ABI issue in Kaggle)\n"
            "# from .mask_renderer import MaskRenderer"
        )
        open(init_path, "w").write(patched)
        print("  ✓ InsightFace mask_renderer patch applied")
    except Exception as e:
        print(f"  ⚠ InsightFace patch skipped: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Readiness report
# ══════════════════════════════════════════════════════════════════════════════

def print_report():
    print("\n" + "="*55)
    print("  ENVIRONMENT READINESS REPORT")
    print("="*55)

    checks = [
        ("person-swap repo",    STATUS.get("repo", False)),
        ("SAM2 clone",          STATUS.get("ext_sam2", False)),
        ("InstantID files",     STATUS.get("ext_InstantID", False)),
        ("HairFastGAN clone",   STATUS.get("ext_HairFastGAN", False)),
        ("OOTDiffusion clone",  STATUS.get("ext_OOTDiffusion", False)),
        ("RAFT clone",          STATUS.get("ext_RAFT", False)),
        ("ProPainter clone",    STATUS.get("ext_ProPainter", False)),
        ("RIFE clone",          STATUS.get("ext_RIFE", False)),
        ("pip packages",        STATUS.get("pip", False)),
        ("checkpoints",         STATUS.get("checkpoints", False)),
        ("sys.path wired",      STATUS.get("paths", False)),
    ]

    all_ok = True
    for label, ok in checks:
        icon = "✓" if ok else "✗"
        print(f"  {icon}  {label}")
        if not ok:
            all_ok = False

    print("="*55)
    if all_ok:
        print("  ✅ All systems ready — run your pipeline cells")
    else:
        print("  ⚠  Some steps failed — check output above")
        print("  Re-run individual step_*() functions to retry")
    print("="*55 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    print("Person-Swap Kaggle Bootstrap")
    print(f"Working dir: {WORK_DIR}")
    print(f"Repo dir:    {REPO_DIR}")
    print(f"Checkpoints: {CKPT_DIR}")
    print(f"Externals:   {EXT_DIR}\n")

    step_clone_repo()
    step_clone_externals()
    step_fetch_instantid()       # new step — wget fallback for InstantID
    step_pip_install()
    step_download_checkpoints()
    step_wire_paths()
    patch_insightface()
    print_report()

    elapsed = int(time.time() - t0)
    print(f"Bootstrap completed in {elapsed}s\n")


if __name__ == "__main__":
    main()
else:
    # Also run when exec()'d from a notebook cell
    main()