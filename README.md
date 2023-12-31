# MDL

This repository is the official implementation of the code for paper [Exploiting Field Dependencies for Learning on Categorical Data](https://arxiv.org/abs/2307.09321).

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
If you find this repository helpful, please consider to cite the following paper:
```
@ARTICLE{10192074,
  author={Li, Zhibin and Koniusz, Piotr and Zhang, Lu and Pagendam, Daniel Edward and Moghadam, Peyman},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Exploiting Field Dependencies for Learning on Categorical Data}, 
  year={2023},
  volume={},
  number={},
  pages={1-14},
  doi={10.1109/TPAMI.2023.3298028}}
```
