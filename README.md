# MDL

This repository is the official implementation of Exploiting Field Dependencies for Learning on Categorical Data.

## Requirements
The code has been tested with:
- Python 3.6.12
- PyTorch 1.10.1
- lmdb 1.1.1
- tqdm 4.57.0


## Training and Evaluation (e.g. on [Avazu](https://www.kaggle.com/c/avazu-ctr-prediction) dataset)
```run
python run_mdl.py   --dataset-path <path_to_data> --inner-step 4 --inner-step-size 1 --lr 0.05 --wdcy 1e-8 --lmbd 100 --ebd-dim 40 
```
## Citation
TBD
