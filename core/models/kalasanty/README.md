[![pipeline status](https://gitlab.com/cheminfIBB/kalasanty/badges/master/pipeline.svg)](https://gitlab.com/cheminfIBB/kalasanty/commits/master)
[![coverage report](https://gitlab.com/cheminfIBB/kalasanty/badges/master/coverage.svg)](https://gitlab.com/cheminfIBB/kalasanty/commits/master)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gl/cheminfIBB%2Fkalasanty/master)


**Kalasanty** is a 3D convolutional neural network that predicts binding sites on protein surface.
It was developed in [keras](https://keras.io/) and trained on the [sc-PDB](https://academic.oup.com/nar/article/43/D1/D399/2439494) database.

The manuscript describing Kalasanty was published in *Scientific Reports* [DOI: 10.1038/s41598-020-61860-z](https://doi.org/10.1038/s41598-020-61860-z).


# Test it online
If you want to quickly test Kalasanty without going through the installation process, you can test it online with [Binder](https://mybinder.org).
Binder will build an environment for you using information in this repository.
Note, that it might take a while, so click the [link](https://mybinder.org/v2/gl/cheminfIBB%2Fkalasanty/master) and go grab a coffee.
*The build might fail if there are too many concurrent users. However, do not get discouraged, please try once more - it will continue from where it timed out.*

When the environment is ready, go to `demo.ipynb`.
If you are not familiar with [Python](https://www.python.org/) and [Jupyter Notebook](https://jupyter.org/), just follow the instructions and run the code (use Cell -> Run Cells in the top panel or press Shift+Enter).
Do not be afraid to play with the code - you cannot break anything.
If the notebook stops working, you can always just launch it again.


# Setup

1) Clone this repository

```bash
git clone https://gitlab.com/cheminfIBB/kalasanty
cd kalasanty
```

2) Create the environment with all dependencies

```bash
# create env
conda env create -f environment.yml -n kalasanty_env
conda activate kalasanty_env

# use GPU if available
if [[ `which nvidia-smi` ]]; then
    conda install -y tensorflow-gpu
fi
```

Note, that you might also need to specify the correct version of CUDA Toolkit (by default conda will install the newest version, which might be incompatible with your drivers), e.g.:.

```bash
if [[ `which nvidia-smi` ]]; then
    conda install -y tensorflow-gpu cudatoolkit=8.0
fi
```

3) Install this package

```bash
pip install .
```

4) Optionally run tests to make sure everything works as expected
```bash
# install pytest
conda install pytest
# run tests
pytest -v
```

# Prepare the data

Proteins should be protonated and charged, and all other molecules (waters,
ions, ligands) should be removed from the structure.

You can test this software using protein structures from the `tests/datasets/` directory.

# Predict druggable pockets for your protein(s)

After preparing your structure(s) as described [above](#prepare-the-data), you can use `predict.py` to make predictions:

```bash
python scripts/predict.py \
  --input tests/datasets/scpdb/*/protein.mol2 tests/datasets/pdbbind/*/*protein.mol2 \
  --output predicted_pockets
```
By default results for each structure are saved in subdirectories named after their parent directories (e.g. results for `tests/datasets/scpdb/3arx/protein.mol2` are saved in `predicted_pockets/3arx/`).
If you organise your data in a different way, you should use `--namedir_pattern` argument to specify way of extracting names from paths.
For more options see `python scripts/predict.py --help`.


# Reproduce this work / train new model

This model was trained using [sc-PDB database](http://bioinfo-pharma.u-strasbg.fr/scPDB/).
If you want to reproduce our work, you must first download the database and calculate features for structures:

```bash
python scripts/prepare_dataset.py \
  --dataset /path/to/sc-PDB/ \
  --output scpdb_dataset.hdf \
  --exclude data/scPDB_blacklist.txt data/scPDB_leakage.txt
```

Note that not all structures are used - we excluded ones that were not readable by Open Babel (`data/scPDB_blacklist.txt`) or appeared in our test set (`data/scPDB_leakage.txt`)

Then you can run training script:

```bash
python scripts/train.py \
  --input scpdb_dataset.hdf \
  --output new_model_scpdb
```

You can also train the model with cross validation:
```bash
python scripts/train.py \
  --input scpdb_dataset.hdf \
  --output new_model_scpdb_fold0 \
  --train_ids data/train_ids_fold0 \
  --test_ids data/test_ids_fold0
```

This script also allows you to fine-tune the model to a different dataset:
```bash
python scripts/train.py \
  --input /path/to/your/dataset.hdf \
  --model data/model_scpdb2017.hdf \
  --output new_model_scpdb
```

For more options see `python scripts/prepare_dataset.py --help` and `python scripts/train.py --help`.
