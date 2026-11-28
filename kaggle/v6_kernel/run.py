# handwriter v6: continue v5-best mixed colour/grey/photo training.
# Same mix as v5 (color_mix=0.70, human_image_prob=0.40). Fresh cosine LR
# peaked at 1.5e-5 (half of v5's 3e-5) with 300-step warmup, because v5's
# cosine had decayed to ~0. Planned 5 epochs at ~95 min/epoch; time cap
# leaves 0.35 h for colour + grey InkML + deepcopy evals. Single T4. No ONNX.
import glob, os, shutil, subprocess, sys, tarfile, time

SMOKE = False
WEEKLY_LEFT_H = 30.00  # fresh weekly quota at launch, 28 Nov 2026 01:00 Europe/Berlin
MARGIN_H = 0.50
# Cap training to ~5 epochs at v5's 95.4 min/epoch, plus a little slack.
# Do not fill the 12 h session or the 30 h week.
MAX_TRAIN_H = 8.30
COLOR_MIX = 0.70
HUMAN_IMAGE_PROB = 0.40
# Fresh continue-FT schedule: half of v5's 3e-5 peak; 300-step warmup.
LR = 1.5e-5
WARMUP = 300
EPOCHS = 5

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

T0 = time.time()
sh("nvidia-smi -L; nproc; free -g; df -h /tmp /kaggle/working")
sh("pip install -q rapidfuzz huggingface_hub")
cands = glob.glob("/kaggle/input/**/v5best__config.json", recursive=True)
assert cands, "v5best not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"
os.makedirs(f"{work}/models/v5best", exist_ok=True)
for f in glob.glob(f"{src}/v5best__*"):
    shutil.copy(f, f"{work}/models/v5best/" + os.path.basename(f).split("__", 1)[1])
code = [p for p in glob.glob(src + "/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if code:
    tarfile.open(code[0]).extractall(work)
else:
    root = os.path.dirname(os.path.dirname(glob.glob(src + "/**/hw/ink.py", recursive=True)[0]))
    for d in ("hw", "scripts"):
        shutil.copytree(f"{root}/{d}", f"{work}/{d}", dirs_exist_ok=True)
assert os.path.exists(f"{work}/hw/data.py"), "stale code archive"
os.chdir(work)

from huggingface_hub import HfApi, hf_hub_download
data = "/tmp/mathwriting"; os.makedirs(data, exist_ok=True)
repo = "deepcopy/MathWriting-human"
for f in HfApi().list_repo_files(repo, repo_type="dataset"):
    if f.endswith(".parquet"):
        p = hf_hub_download(repo, f, repo_type="dataset", local_dir="/tmp/hf")
        os.symlink(p, f"{data}/{os.path.basename(f)}")
print(f"[t] human parquet ready {time.time()-T0:.0f}s", flush=True)

sh("curl -sS -o /tmp/mw.tgz https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz && ls -la /tmp/mw.tgz")
print(f"[t] tgz downloaded {time.time()-T0:.0f}s", flush=True)
rc = sh(f"{sys.executable} scripts/prep_inks.py --tgz /tmp/mw.tgz --out /tmp/inks "
        "--splits train synthetic valid test" + (" --limit 5000" if SMOKE else ""))
assert rc == 0, "prep_inks failed"
os.remove("/tmp/mw.tgz")
sh("ls -la /tmp/inks; free -g")
print(f"[t] inks packed {time.time()-T0:.0f}s", flush=True)

common = (f"{sys.executable} scripts/train.py --data {data} --model models/v5best --out /kaggle/working/run "
          f"--ink-dir /tmp/inks --synth-ratio 1.0 --human-image-prob {HUMAN_IMAGE_PROB} --aug-strength 1.0 "
          f"--color-strokes --color-mix {COLOR_MIX} --eval-plain-images "
          f"--batch-size 32 --eval-batch-size 128 --lr {LR} --workers 3 --max-len 256 --skip-zeroshot ")
if SMOKE:
    cmd = common + f"--train-limit 3000 --valid-limit 200 --test-limit 300 --epochs 2 --max-steps 120 --warmup 20 --beams 1 4"
else:
    g = sh(common.replace("/kaggle/working/run", "/kaggle/working/smoke") +
           "--train-limit 2000 --valid-limit 100 --test-limit 100 --epochs 1 --max-steps 40 --warmup 10 --beams 1 2 --no-save")
    assert g == 0, "smoke gate failed"
    print(f"[t] smoke gate passed {time.time()-T0:.0f}s", flush=True)
    elapsed_h = (time.time() - T0) / 3600
    eval_reserve_h = 0.35  # colour beam 1+4 + grey InkML + deepcopy + full valid
    left_session = 11.2 - elapsed_h - eval_reserve_h  # 12 h session cap
    left_weekly = WEEKLY_LEFT_H - elapsed_h - MARGIN_H - eval_reserve_h
    budget = max(0.5, min(left_session, left_weekly, MAX_TRAIN_H))
    print(f"[t] training budget {budget:.2f} h (weekly left at launch {WEEKLY_LEFT_H:.2f}h, "
          f"elapsed {elapsed_h:.2f}h, margin {MARGIN_H:.2f}h, max_train {MAX_TRAIN_H:.2f}h, "
          f"epochs {EPOCHS}, lr {LR}, warmup {WARMUP}, color_mix {COLOR_MIX}, p_img {HUMAN_IMAGE_PROB})",
          flush=True)
    cmd = (common + f"--valid-limit 2000 --epochs {EPOCHS} --warmup {WARMUP} "
           f"--time-budget-h {budget:.2f} --beams 1 4 --final-full-valid")
rc = sh(cmd)
print(f"[t] training+eval done {time.time()-T0:.0f}s rc={rc}", flush=True)
print(f"[t] all done {time.time()-T0:.0f}s (ONNX skipped)", flush=True)
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
