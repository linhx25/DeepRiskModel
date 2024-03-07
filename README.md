# DeepRiskModel
Deep Risk Model: A Deep Learning Solution for Mining Latent Risk Factors to Improve Covariance Matrix Estimation

Discover risk factors with deep neural networks.

Factor model:

```
r_{1,t} = X_1 @ b_t + u_{1,t}
r_{2,t} = X_2 @ b_t + u_{2,t}

R_1 = [r_{1,t-T}, r_{1,t-T+1}, ..., r_{1,t-1}]' = X_1 @ [b_{t-T}, b_{t-T+1}, ..., b_{t-1}] = X_1 @ B + U_1
R_2 = [r_{2,t-T}, r_{2,t-T+1}, ..., r_{2,t-1}]' = X_2 @ [b_{t-T}, b_{t-T+1}, ..., b_{t-1}] = X_2 @ B + U_2

cov(R_1, R_2) = X_1 @ cov(B, B) @ X_2 + std(U_1) * std(U_2)
```

How to specify `X`:
- Fundamental Risk Model (FRM): `X` is pre-defined by human experts (e.g., Size, Value, Momentum, etc)
- Statistical Risk Model (SRM): `X` is obtained by PCA or Factor Analysis
- Deep Risk Model (DRM): `X` is a learned embedding of input data (thus we are the superset of FRM)


# Methodology

## Model Design

* GAT: cross-sectional information (e.g., return relative to industry)
* RNN: temporal information (e.g., historical momentum)

We use two RNNs to leverage both types of information:
* RNN1: `x -> (GAT) -> x_agg -> (RNN1) -> F1`
* RNN2: `x -> (RNN2) -> F2`

`F1` and `F2` are concatenated as the output risk factors.

## Loss Design

1. R^2
2. Multicollinearity: regularized inverse correlation matrix
3. Stability: multi-task learning

# Experiments

See `run.sh` for more information.

# Citation
```bibtex
@inproceedings{lin2021deep,
  title={Deep risk model: a deep learning solution for mining latent risk factors to improve covariance matrix estimation},
  author={Lin, Hengxu and Zhou, Dong and Liu, Weiqing and Bian, Jiang},
  booktitle={Proceedings of the Second ACM International Conference on AI in Finance},
  pages={1--8},
  year={2021}
}

@article{yang2020qlib,
  title={Qlib: An ai-oriented quantitative investment platform},
  author={Yang, Xiao and Liu, Weiqing and Zhou, Dong and Bian, Jiang and Liu, Tie-Yan},
  journal={arXiv preprint arXiv:2009.11189},
  year={2020}
}
```
