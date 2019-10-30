# SWALP_LR
SWALP for logistic regression on inception V3 features of ImageNet

## References
We use the [SWALP](https://github.com/stevenygd/SWALP) as starter template.

## Dependencies
* CUDA 9.0
* [PyTorch](http://pytorch.org/) version 1.0
* [torchvision](https://github.com/pytorch/vision/)
* [tensorflow](https://www.tensorflow.org/) to use tensorboard

To install other requirements through `$ pip install -r requirements.txt`.

## Dataset

* [DOGFISH](https://drive.google.com/open?id=1qJfVdN9iXZGvmnaqGIHWPZwalx2QWE89) After downloading, we need to copy it to ./features/


## Usage

We provide scripts to run Small-block Block Floating Point experiments on inception v3 features of DOGFISH with Logistic Regression.
Following are scripts to reproduce experimental results.

```bash
seed=100                                      # Specify experiment seed.
bash exp/block_lr_swa.sh DOGFISH ${seed}     # SWALP training on logistic regression with Small-block BFP in DOGFISH

```



