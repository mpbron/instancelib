# Installation
Installation of `instancelib` requires `Python 3.8` or higher.

### 1. Python installation
Install Python on your operating system using the [Python Setup and Usage](https://docs.python.org/3/using/index.html) guide.

### 2. Installing `instancelib`
`instancelib` can be installed:

* _using_ `pip`: `pip3 install` (released on [PyPI](https://pypi.org/project/instancelib/))
* _using_ `uv`: `uv add instancelib` (released on [PyPI](https://pypi.org/project/instancelib/))
* _locally_: cloning the repository and using `uv sync`, or `pip install -e .` (NB. On Ubuntu, you may need to use python3 if python is not available or refers to Python 2.x).

#### Using `pip`
1. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows)
2. Run the command:
    - `pip install instancelib`, or
    - `pip install instancelib`.

```console
user@terminal:~$ pip3 install instancelib
Collecting instancelib
...
Installing collected packages: instancelib
Successfully installed instancelib
```

#### Locally
1. Download the folder from `GitLab/GitHub`:
    - Clone this repository, or 
    - Download it as a `.zip` file and extract it.
2. Open up a `terminal` (Linux / macOS) or `cmd.exe`/`powershell.exe` (Windows) and navigate to the folder you downloaded `instancelib` in.
3. In the main folder (containing the `pyproject.toml` file) run:
    - `uv sync` to create a `.venv` with all dependencies installed, or
    - `pip install -e .` to install it in editable mode into your current environment.

```console
user@terminal:~$ cd ~/instancelib
user@terminal:~/instancelib$ uv sync
Resolved 87 packages in 1.30s
Installed 22 packages in 42ms
 + instancelib==0.5.2 (from file:///home/michielbron/instancelib)
 ...
```
