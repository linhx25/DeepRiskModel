import os
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


def RLS(y, X, R=None, r=None, w=None):
    """Restricted (Weighted) Least Square

    Args:
        y (numpy.array): target value, [#sample]
        X (numpy.array): factors, [#sample, #factor]
        R (numpy.array, optional): constraints, [#factor, #constraint] or [#factor]
        r (numpy.array, optional): target value for constraints, [#constraint]
        w (numpy.array, optional): sample weights, [#sample]

    Returns:
        b (numpy.array): regression coefficients
        resid (numpy.array): regression residual
        tvalues (numpy.array): T-stats for b
        r2 (float): R-Squared

    Reference:
        - [William, 1991] (https://www.jstor.org/stable/2109587)
    """

    # weights
    if w is not None:
        X = X * np.atleast_2d(np.sqrt(w)).T
        y = y * np.sqrt(w)

    # solve
    if R is not None:
        R = np.atleast_2d(R)
        if r is None:
            r = np.zeros(R.shape[0])
        z = np.zeros((len(R), len(R)))
        W = np.block([[X.T @ X, R.T], [R, z]])
        p = np.r_[X.T @ y, r]
        W_inv = np.linalg.pinv(W)
        m = X.shape[1]
        b = W_inv[:m] @ p
        X_inv = W_inv[:m, :m] @ X.T

    else:
        W_inv = np.linalg.pinv(X.T @ X)
        X_inv = W_inv @ X.T
        b = X_inv @ y

    # calc t-value
    resid = y - X @ b
    ss = (resid**2).sum() / (len(resid) - len(b))
    b_var = ss * X_inv @ X_inv.T
    tvalues = b / np.sqrt(np.diag(b_var))

    # calc r2
    r2 = 1 - (resid**2).sum() / (y**2).sum()

    # restore resid
    if w is not None:
        resid /= np.sqrt(w)

    return b, resid, tvalues, r2


def calc_factor_ret(factor_exp, stock_ret, stock_cap, num_group=0,
                    clip_weight=True):
    """estimate factor return

    Note:
        when `num_group > 0`, we expect the factors are ordered by
        [COUNTRY, IND$1, IND$2, ..., IND$num_group, ...]

    Args:
        factor_exp (pandas.DataFrame): factor exposure, [#stock, #factor]
        stock_ret (pandas.Series): stock return, [#stock]
        stock_cap (pandas.Series): stock capitalization, [#stock]
        num_group (int, optional): number of stock group
        clip_weight (bool, optional): whether clip cap weight by 95%

    Returns:
        factor_ret (pandas.Series): factor return, [#factor]
        stock_resid (pandas.Series): stock residual return, [#stock]
        factor_tval (pandas.Series): factor return T-values, [#factor]
        r2 (float): R-Squared
    """

    # store index, output will use this index
    orig_index = factor_exp.index

    # dropna & align index
    factor_exp = factor_exp.dropna()
    stock_ret = stock_ret.reindex(factor_exp.index).fillna(0)
    stock_cap = stock_cap.reindex(factor_exp.index).fillna(0)

    # factor exposure
    X = factor_exp.values

    # sample weights
    w = stock_cap.values**0.5
    if clip_weight:
        q = np.quantile(w, 0.95)
        w[w > q] = q
    w = w / w.sum() * len(w)

    # constraints
    R = None
    if num_group > 0:
        R = np.zeros(X.shape[1])
        slc = slice(1, num_group + 1)
        cap_w = stock_cap.values @ X[:, slc]
        cap_w = cap_w / cap_w.sum() * len(cap_w)
        R[slc] = cap_w

    # target
    y = stock_ret.values

    # regression
    b, resid, tvalues, r2 = RLS(y, X, R, w=w)

    factor_ret = pd.Series(b, index=factor_exp.columns)
    factor_tval = pd.Series(tvalues, index=factor_exp.columns)

    stock_resid = pd.Series(resid, index=factor_exp.index)
    stock_resid = stock_resid.reindex(orig_index)

    return factor_ret, stock_resid, factor_tval, r2


def run(factor_exp, stock_ret, stock_cap, num_group=29, clip_weight=True):

    factor_ret = dict()
    stock_resid = dict()
    factor_tval = dict()
    factor_r2 = dict()

    dates = factor_exp.index.get_level_values(level=0).unique()
    iterator = tqdm(dates)

    for date in iterator:

        iterator.set_description(str(date)[:10])

        ret, resid, tval, r2 = calc_factor_ret(
            factor_exp.loc[date], stock_ret.loc[date],
            stock_cap.loc[date], num_group=num_group,
            clip_weight=clip_weight)

        factor_ret[date] = ret
        stock_resid[date] = resid
        factor_tval[date] = tval
        factor_r2[date] = r2

    factor_ret = pd.DataFrame(factor_ret).T
    stock_resid = pd.DataFrame(stock_resid).T
    factor_tval = pd.DataFrame(factor_tval).T
    factor_r2 = pd.Series(factor_r2)

    return factor_ret, stock_resid, factor_tval, factor_r2


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', default='data')
    parser.add_argument('--num_group', type=int, default=29)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--clip_weight', action='store_true', default=True) # default true

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    factor_exp = pd.read_pickle('data/base.pkl').loc[pd.Timestamp('2015-01-01'):]
    stock_ret = pd.read_pickle('data/ret.pkl').loc[pd.Timestamp('2015-01-01'):]
    stock_cap = pd.read_pickle('data/cap.pkl').loc[pd.Timestamp('2015-01-01'):]

    if os.path.exists(args.outdir + '/pred.pkl'):
        df = pd.read_pickle(args.outdir + '/pred.pkl')
        df.columns = ['RISK%d'%d for d in range(df.shape[1])]
        if args.replace:
            factor_exp = factor_exp.iloc[:, :-10]
        factor_exp[df.columns] = df.loc[pd.Timestamp('2015-01-01'):]

    factor_ret, stock_resid, factor_tval, factor_r2 = run(
        factor_exp, stock_ret, stock_cap,
        num_group=args.num_group, clip_weight=args.clip_weight
    )

    factor_ret.to_pickle(args.outdir + '/factor_ret.pkl')
    stock_resid.to_pickle(args.outdir + '/stock_resid.pkl')
    factor_tval.to_pickle(args.outdir + '/factor_ret_tval.pkl')
    factor_r2.to_pickle(args.outdir + '/factor_ret_r2.pkl')
