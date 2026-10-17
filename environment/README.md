# Environment families

The safe release checker and descriptive table calculation require Python >=3.8 and the standard library only. No Torch, NumPy, SciPy, SB3 or GPU is imported by them.

Recorded research runtimes are distinct from a successfully reconstructed environment:

| Family | Recorded versions | Remaining limit |
| --- | --- | --- |
| Custom PPO, DDQN/D3QN, MovingAI | Python 3.8.10; NumPy 1.24.4; PyTorch 2.3.0+cu121; CUDA 12.1; historical cuDNN 8801 | Incomplete OS/driver/transitive lock; not bound independently for every completed run |
| C51 | Same Python interpreter family pinned by its wrapper; NumPy/PyTorch imports | Completed C51 manifest did not serialize all package versions |
| MaskablePPO | Python 3.10.11; PyTorch 2.8.0+cu128; NumPy 2.2.6; Gymnasium 1.3.0; Stable-Baselines3 2.9.0; SB3-Contrib 2.9.0; cloudpickle 3.1.2 | Separate historical lock, not a tested fresh installation |
| Original inferential analysis | SciPy required, including external-baseline Wilcoxon | Not fully version-locked across analyses; later component analysis recorded SciPy 1.10.1 |
| Figures | Matplotlib used in research | Exact final figure environment and complete source chain unavailable |

Do not merge the Python 3.8/custom PPO and Python 3.10/SB3 families. BARN custom PPO and MovingAI share a package family but have separate data, trained models and protocols.

For a user-managed experimental workspace, the recorded package reconstruction commands are:

```text
py -3.8 -m venv .venv-legacy
.venv-legacy\Scripts\python.exe --version
.venv-legacy\Scripts\python.exe -m pip install numpy==1.24.4
.venv-legacy\Scripts\python.exe -m pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

```text
py -3.10 -m venv .venv-maskable
.venv-maskable\Scripts\python.exe --version
.venv-maskable\Scripts\python.exe -m pip install -r reference/03_Code/external_baselines/environment/requirements-lock.txt
```

Verify CPython patch versions explicitly: the launcher selects a minor version. These installation recipes have **not been executed or validated as a complete fresh-machine reconstruction**. They do not supply datasets, satisfy original hash gates, prove CUDA availability or make training safe to run. The legacy two-package lock is not a complete inference/plot environment. CPU-only substitution, Linux/macOS and alternate CUDA builds are not verified equivalents. No dependencies are needed or installed for the release checks.
