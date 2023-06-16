# Graphically Structured Diffusion Models (GSDM)

[Christian Weilbach](https://whilo.github.io), [William Harvey](https://www.cs.ubc.ca/~wsgh/), [Frank Wood](https://www.cs.ubc.ca/~fwood/)

This codebase provides sparse amortized inference for mixed discrete continuous graphical models. It builds on the diffusion model code in the [DDIM repository](https://github.com/ermongroup/ddim).

You can read our [blog post](https://plai.cs.ubc.ca/2022/11/16/graphically-structured-diffusion-models/) to see demonstrations or dive into our [paper](http://arxiv.org/abs/2209.11633) for more details. 

## Installation

You need Python 3.10 and setup a python environment including PyTorch 1.12 with the requirements listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```
Default experiment settings for each experiment are described in the respective configuration file in `configs`.


## Training commands
We show below all training commands from Figure 6.

### BCMF 

#### GSDM w/ EE
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=True --impose_sparsity sparse --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False`
#### GSDM w/ AE
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=True --impose_sparsity sparse --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False --use_shared_var_positions=True`
#### GSDM w/ IE
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=True --impose_sparsity sparse --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False --use_shared_var_embeds=False`
#### GSDM w/o int.
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=False --impose_sparsity sparse --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False`
#### Non-sparse w/o int.
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=False --impose_sparsity not --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False --use_shared_var_embeds=False`
#### VAEAC
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=False --impose_sparsity not --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False --use_shared_var_embeds=False --vaeac True --ema False --lr 3e-05`
#### Regression + GS
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=True --impose_sparsity sparse --n=16 --m=10 --t=8 --seed=1 --vary_dimensions=False --regression True`
#### Regression
`python main.py --config=configs/bmf.yml --batch_size=8 --sampling_batch_size=5 --fit_intermediate=False --impose_sparsity not --n=16 --m=10 --t=8 --seed=0 --vary_dimensions=False --use_shared_var_embeds=False --regression True`


### Sudoku
#### GSDM w/ EE
`python main.py --config=configs/sudoku.yml --impose_sparsity=sparse --n_epochs=1000 --n_heads=8 --use_shared_var_embeds=True --use_shared_var_positions=False --seed 0`
#### GSDM w/ AE
`python main.py --config=configs/sudoku.yml --impose_sparsity=sparse --n_epochs=1000 --n_heads=8 --use_shared_var_embeds=True --use_shared_var_positions=True --seed 0`
#### GSDM w/ IE
`python main.py --config=configs/sudoku.yml --impose_sparsity=sparse --n_epochs=1000 --n_heads=8 --seed 0`
#### Non-sparse
`python main.py --config=configs/sudoku.yml --impose_sparsity=not --n_epochs=1000 --n_heads=8 --seed 0`
#### VAEAC
`python main.py --config=configs/sudoku.yml --impose_sparsity=sparse --n_epochs=1000 --n_heads=8 --vaeac True --ema False --lr 3e-4 --seed 0`
#### Regression + GS
`python main.py --config=configs/sudoku.yml --impose_sparsity=sparse --n_epochs=1000 --n_heads=8 --use_shared_var_embeds=True --use_shared_var_positions=False --seed 0`
#### Regression
`python main.py --config=configs/sudoku.yml --impose_sparsity=not --n_epochs=1000 --n_heads=8 --regression True --seed 0`


### Sorting
#### GSDM w/ EE
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate true --use_shared_var_embeds true --use_shared_var_positions false --max_epoch_iters 5000 --seed=0`
#### GSDM w/ AE
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate true --use_shared_var_embeds true --use_shared_var_positions true --max_epoch_iters 5000 --seed=0`
#### GSDM w/ IE
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate true --use_shared_var_embeds false --use_shared_var_positions false --max_epoch_iters 5000 --seed=0`
#### GSDM w/o int.
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate false --use_shared_var_embeds false --use_shared_var_positions false --max_epoch_iters 5000 --seed=0`
#### Non-sparse w/o int.
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate false --impose_sparsity not --use_shared_var_embeds false --use_shared_var_positions false --max_epoch_iters 5000 --seed=0`
#### VAEAC
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate true --impose_sparsity not --use_shared_var_embeds false --use_shared_var_positions false --vaeac True --ema False --lr 3e-5 --max_epoch_iters 5000 --seed=0`
#### Regression + GS
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate true --use_shared_var_embeds true --use_shared_var_positions false --regression True --max_epoch_iters 5000 --seed=0`
#### Regression
`python main.py --config=configs/sorting.yml --n_epochs 10000 --n 20 --vary_dimensions false --batch_size 16 --sampling_batch_size 16 --fit_intermediate false --impose_sparsity not --use_shared_var_embeds false --use_shared_var_positions false --regression True --max_epoch_iters 5000 --seed=0`


### Validation

The validation code is automatically run during training, but can also be run after training to recreate evaluation statistics for specific settings, e.g.

```bash
python main.py --batch_size=2 --config=checkpoint_config.yml --eval_path=checkpoint.pth --m=5 --n=5 --sampling_batch_size=2 --t=7
```

## Reference

Please cite this paper if you build on our work.

```bibtex
@misc{weilbachGraphicallyStructuredDiffusion2022,
  title = {Graphically {{Structured Diffusion Models}}},
  author = {Weilbach, Christian and Harvey, William and Wood, Frank},
  year = {2022},
  month = oct,
  number = {arXiv:2210.11633},
  eprint = {2210.11633},
  primaryclass = {cs},
  publisher = {{arXiv}},
  urldate = {2022-10-31},
  abstract = {We introduce a framework for automatically defining and learning deep generative models with problem-specific structure. We tackle problem domains that are more traditionally solved by algorithms such as sorting, constraint satisfaction for Sudoku, and matrix factorization. Concretely, we train diffusion models with an architecture tailored to the problem specification. This problem specification should contain a graphical model describing relationships between variables, and often benefits from explicit representation of subcomputations. Permutation invariances can also be exploited. Across a diverse set of experiments we improve the scaling relationship between problem dimension and our model's performance, in terms of both training time and final accuracy.},
  archiveprefix = {arxiv},
  langid = {english},
  keywords = {Computer Science - Machine Learning,Computer Science - Neural and Evolutionary Computing,Computer Science - Programming Languages,G.3}
}
```


