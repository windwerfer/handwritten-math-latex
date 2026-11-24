# handwriter v3: Texo fine-tune from v2-best on MathWriting human + SYNTHETIC inks with stroke-level
# augmentation (single T4). Evaluates on the official human test set (deepcopy renders, as v1/v2).
import glob, os, shutil, subprocess, sys, tarfile, time

SMOKE = False  # True: short end-to-end check only; False: built-in smoke gate, then full run

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

T0 = time.time()
sh("nvidia-smi -L; nproc; free -g; df -h /tmp /kaggle/working")
sh("pip install -q rapidfuzz huggingface_hub onnx onnxruntime")
cands = glob.glob("/kaggle/input/**/v2best__config.json", recursive=True)
assert cands, "dataset not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"
os.makedirs(f"{work}/models/v2best", exist_ok=True)
for f in glob.glob(f"{src}/v2best__*"):
    shutil.copy(f, f"{work}/models/v2best/" + os.path.basename(f).split("__", 1)[1])
code = [p for p in glob.glob(src + "/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if code:
    tarfile.open(code[0]).extractall(work)
else:  # Kaggle auto-extracted the archive
    root = os.path.dirname(os.path.dirname(glob.glob(src + "/**/hw/ink.py", recursive=True)[0]))
    for d in ("hw", "scripts"):
        shutil.copytree(f"{root}/{d}", f"{work}/{d}", dirs_exist_ok=True)
assert os.path.exists(f"{work}/hw/ink.py"), "stale code archive"
os.chdir(work)

# 1) human images (official splits, pre-rendered) from HF deepcopy/MathWriting-human
from huggingface_hub import HfApi, hf_hub_download
data = "/tmp/mathwriting"; os.makedirs(data, exist_ok=True)
repo = "deepcopy/MathWriting-human"
for f in HfApi().list_repo_files(repo, repo_type="dataset"):
    if f.endswith(".parquet"):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir="/tmp/hf")
        os.symlink(p, f"{data}/{os.path.basename(f)}")
print(f"[t] human parquet ready {time.time()-T0:.0f}s", flush=True)

# 2) official MathWriting-2024 inks (train = human, synthetic) -> packed numpy arrays on disk
sh("curl -sS -o /tmp/mw.tgz https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz && ls -la /tmp/mw.tgz")
print(f"[t] tgz downloaded {time.time()-T0:.0f}s", flush=True)
rc = sh(f"{sys.executable} scripts/prep_inks.py --tgz /tmp/mw.tgz --out /tmp/inks --splits train synthetic"
        + (" --limit 5000" if SMOKE else ""))
assert rc == 0, "prep_inks failed"
os.remove("/tmp/mw.tgz")
sh("ls -la /tmp/inks; free -g")
print(f"[t] inks packed {time.time()-T0:.0f}s", flush=True)

common = (f"{sys.executable} scripts/train.py --data {data} --model models/v2best --out /kaggle/working/run "
          "--ink-dir /tmp/inks --synth-ratio 1.0 --human-image-prob 0.5 --aug-strength 1.0 "
          "--batch-size 32 --eval-batch-size 128 --lr 3e-5 --workers 3 --max-len 256 --skip-zeroshot ")
if SMOKE:
    cmd = common + "--train-limit 3000 --valid-limit 200 --test-limit 300 --epochs 2 --max-steps 120 --warmup 20 --beams 1 4"
else:
    # smoke gate (~3 min): same data/code path on a tiny budget; abort the session if it fails
    g = sh(common.replace("/kaggle/working/run", "/kaggle/working/smoke") +
           "--train-limit 2000 --valid-limit 100 --test-limit 100 --epochs 1 --max-steps 40 --warmup 10 --beams 1 2 --no-save")
    assert g == 0, "smoke gate failed"
    print(f"[t] smoke gate passed {time.time()-T0:.0f}s", flush=True)
    left_h = 11.2 - (time.time() - T0) / 3600 - 0.6   # 12 h Kaggle cap, keep ~35 min for final evals + ONNX
    budget = min(8.5, left_h)
    print(f"[t] training budget {budget:.2f} h", flush=True)
    cmd = common + f"--valid-limit 2000 --epochs 10 --warmup 300 --time-budget-h {budget:.2f} --beams 1 4 --final-full-valid"
rc = sh(cmd)
print(f"[t] training+eval done {time.time()-T0:.0f}s rc={rc}", flush=True)
test = sorted(glob.glob(f"{data}/test-*.parquet"))
if os.path.exists("/kaggle/working/run/best"):
    sh(f"{sys.executable} scripts/export_onnx.py --model /kaggle/working/run/best --out /kaggle/working/onnx "
       f"--test {' '.join(test)} --n {20 if SMOKE else 200}")
print(f"[t] all done {time.time()-T0:.0f}s", flush=True)
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
