# External Validation of IAIA-BL

This codebase contains the codebase for "Multi-site validation of an interpretable model to analyze breast masses."
This codebase is forked from `https://github.com/alinajadebarnett/iaiabl` and the overwhelming majority of the code is from the original repository, and the [original readme](IAIA-BL_README.md) has been moved.
Additions to this repository enable the site-specific analysis presented in the paper.

## Environment Configuration

Create python 3.7.3 environment (`conda env create --file conda-env.yaml`) and activate it (`conda activate iaia-bl-ext`).

Install the dependencies with `pip install -r external-requirements.txt --extra-index-url https://download.pytorch.org/whl/cu92`.

Then fix the dependency conflict with tensorboard and torchaudio by installing `pip install tensorboard==2.1.0 torchaudio==0.7.0 --no-deps --extra-index-url https://download.pytorch.org/whl/cu92`.

Run `Baseline-Logits.ipynb` to test you environment gives you the same logits at each site.

## Site-Specific Notebooks

- `DukeEvaluation.ipynb`: Replicates the results from [1] on Duke's dataset.
- `ICADDataset.ipynb`: Data preparation for the dataset from iCAD.
- `ICADEvaluation.ipynb`: Evaluate models on iCAD Dataset prepared in `ICADDataset.ipynb`.
- `EmoryDataset.ipynb`: Data preparation for the dataset from Emory.
- `EmoryEvaluation.ipynb`: Evaluate models on Emory Dataset prepared in `EmoryDataset.ipynb`.

## Model Weights

## References

[1] Barnett, Alina Jade, et al. "A case-based interpretable deep learning model for classification of mass lesions in digital mammography." Nature Machine Intelligence 3.12 (2021): 1061-1070.