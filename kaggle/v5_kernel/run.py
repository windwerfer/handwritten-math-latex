# handwriter v5: continue from v4-best so one model reads stroke input AND
# plain images (photos/scans). Mix: colour InkML (R=time, G=direction, B=ink),
# greyscale InkML, and official deepcopy raster images. v4 stayed strong on
# grey InkML (63.51% exact) but collapsed on deepcopy photos (25.30%), so
# --human-image-prob is required, not only --color-mix. Same 1:1
# human/synthetic mix + stroke augmentation. Official colour valid/test plus
# a final greyscale InkML and deepcopy-image eval. Single T4. No ONNX.
import glob, os, shutil, subprocess, sys, tarfile, time

SMOKE = False
# Weekly GPU remaining at launch, hours. Leave MARGIN_H unused.
WEEKLY_LEFT_H = 8.08
MARGIN_H = 0.50
COLOR_MIX = 0.70  # fraction of ink renders that keep colour; rest greyscale
HUMAN_IMAGE_PROB = 0.40  # fraction of *human* samples that use official raster images
# Epoch mix at 1:1 human/synth: ~56% colour ink, ~24% grey ink, ~20% photos

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

T0 = time.time()
sh("nvidia-smi -L; nproc; free -g; df -h /tmp /kaggle/working")
sh("pip install -q rapidfuzz huggingface_hub")
cands = glob.glob("/kaggle/input/**/v4best__config.json", recursive=True)
assert cands, "v4best not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"
os.makedirs(f"{work}/models/v4best", exist_ok=True)
for f in glob.glob(f"{src}/v4best__*"):
    shutil.copy(f, f"{work}/models/v4best/" + os.path.basename(f).split("__", 1)[1])
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

common = (f"{sys.executable} scripts/train.py --data {data} --model models/v4best --out /kaggle/working/run "
          f"--ink-dir /tmp/inks --synth-ratio 1.0 --human-image-prob {HUMAN_IMAGE_PROB} --aug-strength 1.0 "
          f"--color-strokes --color-mix {COLOR_MIX} --eval-plain-images "
          "--batch-size 32 --eval-batch-size 128 --lr 3e-5 --workers 3 --max-len 256 --skip-zeroshot ")
if SMOKE:
    cmd = common + "--train-limit 3000 --valid-limit 200 --test-limit 300 --epochs 2 --max-steps 120 --warmup 20 --beams 1 4"
else:
    g = sh(common.replace("/kaggle/working/run", "/kaggle/working/smoke") +
           "--train-limit 2000 --valid-limit 100 --test-limit 100 --epochs 1 --max-steps 40 --warmup 10 --beams 1 2 --no-save")
    assert g == 0, "smoke gate failed"
    print(f"[t] smoke gate passed {time.time()-T0:.0f}s", flush=True)
    elapsed_h = (time.time() - T0) / 3600
    eval_reserve_h = 0.35  # colour beam 1+4 + grey InkML + deepcopy + full valid
    left_session = 11.2 - elapsed_h - eval_reserve_h  # 12 h session cap
    left_weekly = WEEKLY_LEFT_H - elapsed_h - MARGIN_H - eval_reserve_h
    budget = max(0.5, min(left_session, left_weekly))
    print(f"[t] training budget {budget:.2f} h (weekly left at launch {WEEKLY_LEFT_H:.2f}h, "
          f"elapsed {elapsed_h:.2f}h, margin {MARGIN_H:.2f}h, color_mix {COLOR_MIX}, p_img {HUMAN_IMAGE_PROB})", flush=True)
    cmd = common + f"--valid-limit 2000 --epochs 10 --warmup 200 --time-budget-h {budget:.2f} --beams 1 4 --final-full-valid"
rc = sh(cmd)
print(f"[t] training+eval done {time.time()-T0:.0f}s rc={rc}", flush=True)
print(f"[t] all done {time.time()-T0:.0f}s (ONNX skipped)", flush=True)
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
