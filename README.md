# sol_polygnn

This repository contains the training code and weights for a polymer-solvent ML model presented in the companion paper, [AI-assisted discovery of high-temperature dielectrics for energy storage](https://www.nature.com/articles/s41467-024-50413-x).

## Installation
This repository is currently set up to run on Linux machines with CUDA 10.2. Please raise a GitHub issue if you want to use this repo with a different configuration. Otherwise, follow these steps for installation:

1. Install [poetry](https://python-poetry.org/) on your machine.
2. If Python3.9 is installed on your machine skip to step 3, if not you will need to install it. There are many ways to do this, one option is detailed below:
    * Install [Homebrew](https://brew.sh/) on your machine.
    * Run `brew install python@3.9`. Take note of the path to the python executable.
3. Clone this repo on your machine.
4. Open a terminal at the root directory of this repository.
5. Run `poetry env use /path/to/python3.9/executable`.
7. Run `poetry install`.
8. Run `poetry run poe torch-linux_win-cuda102`.
9. Run `poetry run poe pyg-linux-cuda102`.

## Usage
### `example.py`
The file `example.py` contains example code that illustrates how to use the ML model to predict polymer-solvent compatibility. In particular, the model is used to predict if trichlorobenzene is a "bad_solvent", "medium_solvent", or "good_solvent" for polyethylene. To run the file, execute `poetry run python example.py`.

## Questions
I ([@rishigurnani](https://github.com/rishigurnani)) am more than happy to answer any questions about this codebase. If you encounter any troubles, please open a new Issue in the "Issues" tab and I will promptly respond. In addition, if you discover any bugs or have any suggestions to improve the codebase (documentation, features, etc.) please also open a new Issue. This is the power of open source!

## License
This repository is protected under a General Public Use License Agreement, the details of which can be found in `GT Open Source General Use License.pdf`.

## Reproducibility
The version of this codebase that was used in the companion paper is v0.3.0.
