# Python, PyTorch, and Images
The purpose of this repository is to explore working with images when using Python and PyTorch.

## Instalation
I'm currently running this on an M1 MacBook. If you are on a different platform, follow the standard install instructions.

- Install miniforge from https://github.com/conda-forge/miniforge#miniforge3.
- Configure miniforge.  Create similar to:
```
conda create -n torch_macos python=3.8
conda activate torch_macos
```
I went with python 3.8 to match apple's TensorFlow version.

- Install PyTorch and TorchVision (I havent' built audio yet)
```
  conda install pytorch torchvision torchaudio -c pytorch -c=conda-forge
```
I needed add `-c=conda-forge` to deal with a likely python/pytorch mismatch.


- Using Pillow
- Using Imageio




The file `horse.jpg` is taking from https://github.com/deep-learning-with-pytorch/dlwpt-code.
