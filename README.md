# PINN-PDE-compatibility

This repository is a supplementary material of the following paper:

Kuangdai Leng, and Jeyan Thiyagalingam. On the compatibility between neural networks and partial differential equations for physics-informed learning. arXiv preprint [arXiv:2212.00270](https://arxiv.org/abs/2212.00270), 2022.

```BibTeX
@article{leng2022on,
  title={On the compatibility between neural networks and partial differential equations for physics-informed learning},
  author={Leng, Kuangdai and Thiyagalingam, Jeyan},
  journal={arXiv preprint arXiv:2212.00270},
  year={2022}
}
```

### Installation
[PyTorch](https://pytorch.org/get-started/locally/) is the only dependency. You do NOT need a GPU to run the scripts provided here.

### The scripts
Three standalone Python scripts are provided:

* `relu_causes_zero_hessian.py` verifies that a multilayer
perceptron (MLP) only with ReLU-like activation functions will always lead to a 
vanished Hessian;
* `piecewise_linear_by_relu.py` plots the piecewise linear functions generated by some random ReLU-based MLPs, as a support to Figure 2b in the paper;
* `out_layer_hyperplane.py` implements the *out-layer-hyperplane* in an MLP for a 2D inhomogeneous wave equation on a membrane: $u_{xx} + u_{yy} − u_{tt} − u_{t} = \sin(x + y − t)$, following Algorithm 1 in the paper. This verifies that an MLP equipped with the out-layer-hyperplane can always satisfy a given linear PDE at every point.


