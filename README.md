# PINN-PDE-compatibility

This repository is a supplementary material of the following paper:

*Kuangdai Leng, and Jeyan Thiyagalingam. On the compatibility between a neural network and a partial differential equation for physics-informed learning. arXiv preprint [arXiv:????.?????](????), 2022.*

BibTeX:
```BibTeX
@article{leng2022on,
  title={On the compatibility between a neural network and a partial differential equation for physics-informed learning},
  author={Leng, Kuangdai and Thiyagalingam, Jeyan},
  journal={arXiv preprint arXiv:????.?????},
  year={2022}
}
```

### Installation
`Pytorch` is the only dependency of this code. Installation of `Pytorch` can be found 
[here](https://pytorch.org/get-started/locally/). You do NOT need a GPU to run the scripts provided here.

### The scripts
Three Python scripts are provided here.

* `relu_causes_zero_hessian.py` verifies that a multilayer
perceptron (MLP) only with ReLU-like activation functions will always lead to a 
vanished Hessian;
* `piecewise_linear_by_relu.py` plots the piecewise linear functions generated by a few ReLU-based MLPs, as a support to Figure 2b in the paper;
* `out_layer_hyperplane.py` implements the *out-layer-hyperplane* in an MLP for a 2D inhomogeneous wave equation on a membrane, $u_{xx} + u_{yy} − u_{tt} − u_{t} = \sin(x + y − t)$, following Algorithm 1 in the paper. This verifies that an MLP equipped with the out-layer-hyperplane can always satisfy a given PDE at every point.


