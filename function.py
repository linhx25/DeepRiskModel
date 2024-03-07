import torch


def zscore(x, w=None, mask_w=False):
    # robust zscore
    med = torch.median(x, dim=0, keepdim=True)[0]
    mad = torch.median((x - med).abs(), dim=0, keepdim=True)[0] * 1.4826
    x = torch.min(torch.max(x, med - mad * 3), med + mad * 3)
    # normal zscore
    if w is not None:
        if mask_w:
            mask = ~torch.isnan(w)
            mean = x[mask].T @ w[mask] / w[mask].sum()
        else:
            mean = x.T @ w / w.sum()
    else:
        mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, unbiased=False, keepdim=True) + 1e-12
    return (x - mean) / std


def RLS(y, X, R=None, weights=None):
    """Restricted Least Square"""
    # add weights
    if len(y.shape) == 1:
        y = y[:, None]
    if weights is not None:
        w = weights.sqrt()[:, None] # atleast_2d
        X = X * w
        y = y * w
    # solve RLS
    if R is not None:
        R = R.view(1, -1)
        _ = torch.zeros(len(R), len(R), device=y.device)
        W = torch.cat([torch.cat([X.T@X, R.T], dim=1),
                       torch.cat([R, _], dim=1)], dim=0)
        _ = torch.zeros(1, y.shape[1], device=y.device)
        p = torch.cat([X.T @ y, _], dim=0)
        W_inv = torch.inverse(W)
        m = X.shape[1]
        b = W_inv[:m] @ p
        X_inv = W_inv[:m, :m] @ X.T
    else:
        W_inv = torch.inverse(X.T @ X)
        X_inv = W_inv @ X.T
        b = X_inv @ y
    # calc t-value
    eps = (y - X @ b)**2
    ss = eps.sum(dim=0) / (len(eps) - len(b))
    b_var = torch.diag(X_inv @ X_inv.T)[:, None] * ss
    tvalues = b / b_var.sqrt()
    # calc r2
    r2 = 1 - eps.sum(dim=0) / (y**2).sum(dim=0)
    # tw = torch.arange(len(r2), 0., -1, device=r2.device)  # linear decay
    # tw /= tw.sum()
    # r2 = r2 @ tw
    r2 = r2.mean()
    # calc norm
    norm = torch.trace(W_inv) * len(X) / X.shape[1]
    return b.squeeze(), tvalues.squeeze(), r2, norm


def regression(pred, label, cap, add_intercept=True, clip_weight=True):
    if len(label.shape) == 1:
        label = label[:, None]  # extend to 2d
    mask = ~(torch.isnan(label[:, 0]) | torch.isnan(cap))
    pred, label, cap = pred[mask], label[mask], cap[mask]
    label = torch.nan_to_num(label)
    X = zscore(pred, cap)
    # norm = (X.T @ X / len(X) - torch.eye(X.shape[1], device=X.device)).pow(2).mean()
    if add_intercept:
        const = torch.ones(len(X), 1, device=X.device)
        X = torch.cat([const, X], dim=1)
    y = label
    weights = torch.sqrt(cap)
    if clip_weight:
        weights.clamp_(max=torch.quantile(weights, 0.95))
    weights.div_(weights.sum()).mul_(len(weights))
    return RLS(y, X, weights=weights) # + (norm,)
