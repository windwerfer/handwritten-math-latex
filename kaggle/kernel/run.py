# handwriter v2: Texo fine-tune on full MathWriting (human) — Kaggle GPU kernel.
import glob, os, shutil, subprocess, sys, tarfile

SMOKE = False  # True: short pipeline check (~10 min); False: full run

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

sh("nvidia-smi; nproc; free -g; df -h /tmp /kaggle/working")
sh("pip install -q rapidfuzz huggingface_hub")

cands = glob.glob("/kaggle/input/**/texo__config.json", recursive=True)
assert cands, "dataset not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"; os.makedirs(work + "/models/texo", exist_ok=True)
for f in glob.glob(src + "/texo__*"):
    shutil.copy(f, work + "/models/texo/" + os.path.basename(f).split("__", 1)[1])
code = [p for p in glob.glob(src + "/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if code:
    tarfile.open(code[0]).extractall(work)
else:  # Kaggle auto-extracted the archive
    root = os.path.dirname(os.path.dirname(glob.glob(src + "/**/hw/model.py", recursive=True)[0]))
    for d in ("hw", "scripts"):
        shutil.copytree(f"{root}/{d}", f"{work}/{d}", dirs_exist_ok=True)
sh(f"ls -R {work} | head -40; grep -c normalize {work}/hw/latex_tok.py")
os.chdir(work)

# MathWriting-2024 human inks, official splits, pre-rendered: HF deepcopy/MathWriting-human
from huggingface_hub import HfApi, hf_hub_download
data = "/tmp/mathwriting"; os.makedirs(data, exist_ok=True)
repo = "deepcopy/MathWriting-human"
for f in HfApi().list_repo_files(repo, repo_type="dataset"):
    if f.endswith(".parquet"):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir="/tmp/hf")
        os.symlink(p, f"{data}/{os.path.basename(f)}")
sh(f"ls -la {data}")

common = (f"{sys.executable} scripts/train.py --data {data} --model models/texo --out /kaggle/working/run "
          "--batch-size 64 --eval-batch-size 128 --lr 5e-5 --workers 4 --max-len 256 --data-parallel ")
if SMOKE:
    cmd = common + "--train-limit 20000 --valid-limit 200 --test-limit 300 --epochs 1 --max-steps 300 --warmup 50 --beams 1 4"
else:
    cmd = common + ("--valid-limit 2000 --epochs 10 --warmup 500 --time-budget-h 9.5 "
                    "--beams 1 4 --final-full-valid --skip-zeroshot")
rc = sh(cmd)
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
