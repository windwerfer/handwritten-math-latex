# handwriter v2-eval: full MathWriting test (7644) for Texo zero-shot, v1 and v2-best checkpoints,
# canonical (hw.normalize) scoring, greedy + beam 4 for v2.
import glob, os, shutil, subprocess, sys, tarfile

def sh(c):
    print("+", c, flush=True); return subprocess.call(c, shell=True)

sh("nvidia-smi -L; nproc; free -g")
sh("pip install -q rapidfuzz huggingface_hub")
cands = glob.glob("/kaggle/input/**/v2best__config.json", recursive=True)
assert cands, "dataset not mounted: " + str(glob.glob("/kaggle/input/**", recursive=True)[:50])
src = os.path.dirname(cands[0]); print("dataset dir", src)
work = "/kaggle/working/handwriter"
for pre in ("texo", "v1", "v2best"):
    os.makedirs(f"{work}/models/{pre}", exist_ok=True)
    for f in glob.glob(f"{src}/{pre}__*"):
        shutil.copy(f, f"{work}/models/{pre}/" + os.path.basename(f).split("__", 1)[1])
code = [p for p in glob.glob(src + "/**/code.tar.gz", recursive=True) if os.path.isfile(p)]
if code:
    tarfile.open(code[0]).extractall(work)
else:  # Kaggle auto-extracted the archive
    root = os.path.dirname(os.path.dirname(glob.glob(src + "/**/hw/model.py", recursive=True)[0]))
    for d in ("hw", "scripts"):
        shutil.copytree(f"{root}/{d}", f"{work}/{d}", dirs_exist_ok=True)
os.chdir(work)
from huggingface_hub import HfApi, hf_hub_download
repo = "deepcopy/MathWriting-human"
test = [hf_hub_download(repo, f, repo_type="dataset", local_dir="/tmp/hf")
        for f in HfApi().list_repo_files(repo, repo_type="dataset") if f.startswith("data/test-")]
rc = sh(f"{sys.executable} scripts/eval.py --test {' '.join(test)} --out /kaggle/working/eval "
        "--eval-batch-size 128 --workers 3 "
        "--model texo_zeroshot=models/texo:1 --model v1=models/v1:1 --model v2best=models/v2best:1,4")
shutil.rmtree(work, ignore_errors=True)
sys.exit(rc)
