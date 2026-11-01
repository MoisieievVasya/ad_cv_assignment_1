# %% [markdown]
# # Colab / Kaggle runner
# Runs any experiment config of this repo on a cloud GPU with the same code, data pipeline and folds as
# local runs, then sends the results (`experiments/<name>/`) back to GitHub on a `results/...` branch.
#
# **Colab secrets** (key icon in the left panel, enable notebook access):
# `KAGGLE_API_TOKEN` (Kaggle → Settings → API token) and `GH_TOKEN` (GitHub fine-grained token with
# *Contents: read & write* on this repository only). Runtime → Change runtime type → GPU (A100 / L4 / T4).

# %%
CONFIG = "configs/p2/vit.yaml"   # which grid to run
ONLY = ""                         # one experiment name from the file, or "" for all
REPO = "MoisieievVasya/ad_cv_assignment_1"

# %%
import base64, os, subprocess
try:
    from google.colab import userdata
    secret = userdata.get
    WORK = "/content"
except ImportError:  # Kaggle: Add-ons → Secrets
    from kaggle_secrets import UserSecretsClient
    secret = UserSecretsClient().get_secret
    WORK = "/kaggle/working"
os.environ["KAGGLE_API_TOKEN"] = secret("KAGGLE_API_TOKEN")
GH = secret("GH_TOKEN").strip()
# The token goes into an HTTP header, never into a URL or a command line, and is masked in errors.
AUTH = "AUTHORIZATION: basic " + base64.b64encode(f"x-access-token:{GH}".encode()).decode()

def git(*args, cwd=None):
    r = subprocess.run(["git", "-c", f"http.https://github.com/.extraheader={AUTH}", *args],
                       cwd=cwd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).replace(GH, "***").replace(AUTH, "***")
    if r.returncode:
        raise RuntimeError(f"git {args[0]} failed:\n{out}")
    return out

os.chdir(WORK)
if not os.path.exists("repo"):
    git("clone", "-q", f"https://github.com/{REPO}.git", "repo")
os.chdir("repo")
git("pull", "-q")
print(git("log", "--oneline", "-1"))
!nvidia-smi --query-gpu=name,memory.total --format=csv

# %%
# Cloud images ship their own CUDA torch; install only what is missing, then the package without deps.
!pip install -q "timm==1.0.30" "peft==0.21.2" kaggle six
!pip install -q -e . --no-deps

# %%
if not os.path.exists("data/raw/Training_labels_updated.csv"):
    !kaggle competitions download soil-grain-size-from-photos -p data/raw -q
    !cd data/raw && unzip -q -o *.zip && rm -f *.zip
!python scripts/prepare_data.py   # rebuilds the 4.5 px/mm cache; folds.csv comes from git, unchanged
!git status --short data/        # must be empty: same photo index and folds as the local runs

# %%
only = f"--only {ONLY}" if ONLY else ""
!python scripts/train.py {CONFIG} {only}

# %%
branch = "results/" + os.path.splitext(os.path.basename(CONFIG))[0] + (f"-{ONLY}" if ONLY else "")
git("config", "user.email", "colab@runner"); git("config", "user.name", "colab-runner")
git("checkout", "-q", "-B", branch)
git("add", "experiments")
git("commit", "-q", "-m", f"Results of {CONFIG} {ONLY} (cloud run)")
git("push", "-q", "-f", "origin", branch)
print("pushed to", branch)
