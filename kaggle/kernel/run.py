# handwriter: Texo fine-tune on MathWriting subset (Kaggle GPU kernel).
import glob, os, shutil, subprocess, sys, tarfile

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

sh("nvidia-smi")
# Pin transformers to the version Texo was built with (needs internet; fall back to preinstalled).
if sh("pip install -q 'transformers==4.40.0' 'tokenizers<0.20' rapidfuzz") != 0:
    sh("pip install -q rapidfuzz")

cands = glob.glob("/kaggle/input/**/texo__config.json", recursive=True)
assert cands, "dataset not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"; os.makedirs(work + "/models/texo", exist_ok=True)
for f in glob.glob(src + "/texo__*"):
    shutil.copy(f, work + "/models/texo/" + os.path.basename(f).split("__", 1)[1])
code = glob.glob(src + "/code.tar.gz")
if code:
    tarfile.open(code[0]).extractall(work)
else:  # Kaggle may have auto-extracted the archive
    for d in ("hw", "scripts"):
        p = glob.glob(src + "/**/" + d, recursive=True)[0]; shutil.copytree(p, work + "/" + d)
os.chdir(work)
os.makedirs("data", exist_ok=True)
if os.path.exists("data/subset"): shutil.rmtree("data/subset")
os.symlink(src, "data/subset")
# test_full = all 7644 official MathWriting test inks
os.makedirs("data/full_test", exist_ok=True)
for s, d in [("train", "train"), ("valid", "valid"), ("test_full", "test")]:
    p = f"data/full_test/{d}.parquet"
    if not os.path.exists(p): os.symlink(f"{src}/{s}.parquet", p)
rc = sh(f"{sys.executable} scripts/train.py --data data/full_test --model models/texo "
        "--out /kaggle/working/run --epochs 5 --batch-size 32 --eval-batch-size 64 "
        "--lr 5e-5 --warmup 100 --workers 2 --max-len 256")
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
