# handwriter v4: continue from v3-best with stroke-order/direction encoded as RGB channels
# in InkML renders (R=time, G=direction, B=ink). Official valid/test are also rendered
# from InkML (deterministic). Same 1:1 human/synthetic mix + stroke augmentation as v3.
# Texo's HGNetv2 stem is already 3-channel; we do not widen it. Single T4.
import glob, os, shutil, subprocess, sys, tarfile, time

SMOKE = False  # True: short end-to-end check only; False: built-in smoke gate, then full run

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

T0 = time.time()
sh("nvidia-smi -L; nproc; free -g; df -h /tmp /kaggle/working")
sh("pip install -q rapidfuzz huggingface_hub")
cands = glob.glob("/kaggle/input/**/v3best__config.json", recursive=True)
assert cands, "v3best not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"
os.makedirs(f"{work}/models/v3best", exist_ok=True)
for f in glob.glob(f"{src}/v3best__*"):
    shutil.copy(f, f"{work}/models/v3best/" + os.path.basename(f).split("__", 1)[1])
code = [p for p in glob.glob(src + "/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if code:
    tarfile.open(code[0]).extractall(work)
else:
    root = os.path.dirname(os.path.dirname(glob.glob(src + "/**/hw/ink.py", recursive=True)[0]))
    for d in ("hw", "scripts"):
        shutil.copytree(f"{root}/{d}", f"{work}/{d}", dirs_exist_ok=True)
assert os.path.exists(f"{work}/hw/ink.py"), "stale code archive"
os.chdir(work)

# 1) official split parquet (labels + sample_ids; images unused under --color-strokes)
from huggingface_hub import HfApi, hf_hub_download
data = "/tmp/mathwriting"; os.makedirs(data, exist_ok=True)
repo = "deepcopy/MathWriting-human"
for f in HfApi().list_repo_files(repo, repo_type="dataset"):
    if f.endswith(".parquet"):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir="/tmp/hf")
        os.symlink(p, f"{data}/{os.path.basename(f)}")
print(f"[t] human parquet ready {time.time()-T0:.0f}s", flush=True)

# 2) official MathWriting-2024 inks, including valid + test (needed for colour eval)
sh("curl -sS -o /tmp/mw.tgz https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz && ls -la /tmp/mw.tgz")
print(f"[t] tgz downloaded {time.time()-T0:.0f}s", flush=True)
rc = sh(f"{sys.executable} scripts/prep_inks.py --tgz /tmp/mw.tgz --out /tmp/inks "
        "--splits train synthetic valid test" + (" --limit 5000" if SMOKE else ""))
assert rc == 0, "prep_inks failed"
os.remove("/tmp/mw.tgz")
sh("ls -la /tmp/inks; free -g")
print(f"[t] inks packed {time.time()-T0:.0f}s", flush=True)

common = (f"{sys.executable} scripts/train.py --data {data} --model models/v3best --out /kaggle/working/run "
          "--ink-dir /tmp/inks --synth-ratio 1.0 --human-image-prob 0.0 --aug-strength 1.0 --color-strokes "
          "--batch-size 32 --eval-batch-size 128 --lr 3e-5 --workers 3 --max-len 256 --skip-zeroshot ")
if SMOKE:
    cmd = common + "--train-limit 3000 --valid-limit 200 --test-limit 300 --epochs 2 --max-steps 120 --warmup 20 --beams 1 4"
else:
    g = sh(common.replace("/kaggle/working/run", "/kaggle/working/smoke") +
           "--train-limit 2000 --valid-limit 100 --test-limit 100 --epochs 1 --max-steps 40 --warmup 10 --beams 1 2 --no-save")
    assert g == 0, "smoke gate failed"
    print(f"[t] smoke gate passed {time.time()-T0:.0f}s", flush=True)
    # weekly quota ~14 h; cap training at 7 h so about 5 h remain after this session's overhead
    left_h = 11.2 - (time.time() - T0) / 3600 - 0.6  # 12 h session cap, keep ~35 min for final evals
    budget = min(7.0, left_h)
    print(f"[t] training budget {budget:.2f} h", flush=True)
    cmd = common + f"--valid-limit 2000 --epochs 10 --warmup 300 --time-budget-h {budget:.2f} --beams 1 4 --final-full-valid"
rc = sh(cmd)
print(f"[t] training+eval done {time.time()-T0:.0f}s rc={rc}", flush=True)
# ONNX export skipped here: parity would need colour InkML test images; do it after we keep the run.
print(f"[t] all done {time.time()-T0:.0f}s", flush=True)
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
