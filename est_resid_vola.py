import os
import argparse
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from utils import get_exp_weights, weighted_var, weighted_mean, RLS


def calc_resid_vola(stock_resid, tau=84, tau_nw=252, lags_nw=5):
    """calculate residual volatility

    Args:
        stock_resid (pandas.DataFrame): residual return, [#date, #stock]
        tau (int, optional): exponential decay halflife
        lags_nw (int, optional): delay lags for Newey-West adjustment
            when `lags_nw=0` Newey-West won't be used
        tau_nw (int, optional): exponential decay halflife for \
            Newey-West adjustment

    Returns:
        pandas.Series: residual volatility

    Reference:
        - Newey, W. K., & West, K. D. (1986). A simple, positive semi-definite, \
          heteroskedasticity and autocorrelation consistent covariance matrix.
    """
    # get shape
    Tn, Fn = stock_resid.shape

    # get exponential decay weights
    w = get_exp_weights(tau, Tn)
    w_nw = get_exp_weights(tau_nw, Tn)

    # calculate weighted variance
    var = weighted_var(stock_resid.values, w)

    # Newey-West adjustment for autocorrelation
    for d in range(1, lags_nw + 1):
        var_nw = weighted_var(stock_resid.values, w_nw, lags=d)
        var += (1 - d / (lags_nw + 1.)) * 2 * var_nw

    # transform into series
    var = pd.Series(var, index=stock_resid.columns)

    # fix var with small windows
    var = var.where(stock_resid.count() >= 30)

    # TODO: fix var < 0

    return np.sqrt(var)


def calc_blend_coef(stock_resid):
    """calculate the blending coefficient

    Args:
        stock_resid (pandas.DataFrame): residual returns, [#date, #stock]

    Returns:
        pandas.Series: blending coefficient (gamma), [#stock]

    References:
        - Briner, Beat, Rachael Smith, and Paul Ward. 2009.
          "The Barra European Equity Model (EUE3)."
          Research Notes. P31-P32
    """

    sigma_robust = stock_resid.quantile(0.75) - stock_resid.quantile(0.25)
    sigma_robust /= 1.35

    # NOTE: clip is too slow, skip
    # bound = sigma_robust * 10
    # stock_resid = stock_resid.clip(lower=-bound, upper=bound, axis=1)
    sigma = stock_resid.std(ddof=0)

    Z = abs(sigma / sigma_robust - 1)

    h = stock_resid.count()

    gamma = np.minimum(1, np.maximum(0, (h - 60) / 120))
    gamma *= np.minimum(1, np.maximum(0, np.exp(1 - Z)))

    return gamma


def adj_resid_vola_str(resid_vola, factor_exp, stock_cap, stock_resid,
                       gamma, clip_weight=True):
    """adjust residual volatility by structure model

    Args:
        resid_vola (pandas.Series): residual volatility, [#stock]
        factor_exp (pandas.DataFrame): factor exposure, [#stock, #factor]
        stock_cap (pandas.Series): stock capitalization, [#stock]
        stock_resid (pandas.Series): stock residual returns, [#stock]
        gamma (pandas.Series): blending coefficient, [#stock]
        clip_weight (bool, optional): whether clip cap weight by 95%

    Returns:
        pandas.Series: residual volatility using
            structure model [#stock]

    References:
        - Menchero, J., Orr, D. J., & Wang, J. (2011). \
          The Barra US equity model (USE4), methodology notes. \
          MSCI Barra., P30-P31
    """

    # align index to residual vola
    factor_exp = factor_exp.reindex(resid_vola.index)
    stock_cap = stock_cap.reindex(resid_vola.index)
    stock_resid = stock_resid.reindex(resid_vola.index)

    # dropna
    m1 = ~factor_exp.isna().any(axis=1).values
    m2 = ~resid_vola.isna().values
    m3 = ~stock_cap.isna().values
    m4 = ~stock_resid.isna().values
    shared_index = factor_exp.index[m1].intersection(resid_vola.index[m2]).intersection(
        stock_cap.index[m3]).intersection(stock_resid.index[m4])

    X = factor_exp.reindex(shared_index).values
    X = np.c_[X, stock_resid.reindex(shared_index).abs().values]
    y = np.log(resid_vola.reindex(shared_index).values)

    # cap weights
    w = stock_cap.reindex(shared_index).values**0.5
    if clip_weight:
        q = np.quantile(w, 0.95)
        w[w > q] = q
    w = w / w.sum() * len(w)

    # weighted regression
    b, resid, tvalues, r2 = RLS(y, X, w=w)
    if r2 < 0.8:
        print('WARN: structure model has low R^2 (%.3f)'%r2)

    # predict
    resid_vola_str = pd.Series(np.exp(X @ b), index=shared_index)

    # scale by ratio
    ratio = resid_vola.reindex(shared_index) / resid_vola_str
    E0 = weighted_mean(ratio, w)

    resid_vola_str *= E0

    # adjust
    resid_vola_adj = gamma * resid_vola + (1 - gamma) * resid_vola_str

    # use TS when STR is not available
    resid_vola_adj.fillna(resid_vola, inplace=True)

    return resid_vola_adj


def adj_resid_vola_bayes(resid_vola, stock_cap, q=0.1):
    """adjust residual volatility by bayesian shrinkage

    Args:
        resid_vola (pandas.Series): residual volatility, [#stock]
        stock_cap (pandas.Series): stock capitalization, [#stock]
        q (float): shrinkage parameter

    Returns:
        pandas.Series: adjusted residual volatility, [#stock]

    References:
        - Menchero, J., Orr, D. J., & Wang, J. (2011). \
          The Barra US equity model (USE4), methodology notes. \
          MSCI Barra., P30-P31
    """
    stock_cap = stock_cap.loc[resid_vola.index]
    decile = np.floor(stock_cap.rank(pct=True).mul(9.99))

    for d in range(10):

        mask = (decile == d)

        vola = resid_vola[mask]
        cap = stock_cap[mask]

        # will report error
        # pd.Series cannot be reshaped, need to change into numpy.ndarray first
        target = weighted_mean(vola.values, cap.values)
        delta = np.sqrt(weighted_var(vola, cap))
        diff = (vola - target).abs()

        v = q * diff / (delta + q * diff)

        resid_vola.loc[mask] = v * target + (1 - v) * vola

    return resid_vola


def adj_resid_vola_vra(resid_vola, bias_stats, tau=42):
    """adjust factor covariance by Volatility Regime Adjustment

    Args:
        resid_vola (pandas.Series): residual volatility
        bias_stats (list): history bias statistic B^2
        tau (int): exponential decay halflife

    Returns:
        resid_vola (pandas.Series): adjusted residual volatility
        lamb (float): volatility adjustment multiplier

    References:
        - Menchero, J., Orr, D. J., & Wang, J. (2011). \
          The Barra US equity model (USE4), methodology notes. \
          MSCI Barra., P31-P32
    """

    # calc factor volatility multiplier
    lamb = 1.0
    if len(bias_stats) >= tau:
        w = get_exp_weights(tau, len(bias_stats))
        lamb = np.average(bias_stats, weights=w) # NOTE: \lambda^2
        lamb = np.sqrt(lamb)

    # adjust covariance
    resid_vola *= lamb

    return resid_vola, lamb


def run(stock_resid, stock_cap, factor_exp, tau_vola=84,
        lags_nw=5, tau_nw=252, tau_vra=42, gamma_T=180,
        max_T=None, adj_str=True, adj_bayes=True,
        adj_vra=True, clip_weight=True):

    min_T = max([tau_vola, tau_nw, tau_vra]) # ensure 50%
    if max_T is None:
        max_T = int(min_T * np.log(1 - 0.95) / np.log(0.5)) # ensure 95%

    bias_stats = []
    multipliers = []

    gammas = dict()

    res = dict()
    iterator = tqdm(range(min_T, len(stock_resid)))

    for i in iterator:

        date = stock_resid.index[i]
        iterator.set_description(str(date)[:10])

        # calc volatility
        slc = slice(max(i - max_T, 0), i) # NOTE: i is not included
        vola = calc_resid_vola(stock_resid.iloc[slc], tau=tau_vola,
                               tau_nw=tau_nw, lags_nw=lags_nw)

        # adjust volatility

        gammas[date] = pd.Series(1.0, index=stock_resid.columns)

        if adj_str:

            # calc blending coefficient
            gamma = calc_blend_coef(stock_resid.iloc[i-gamma_T:i])
            gammas[date] = gamma

            # adjust
            vola = adj_resid_vola_str(
                vola, factor_exp.loc[date], stock_cap.iloc[i],
                stock_resid.iloc[i-1], gamma, clip_weight=clip_weight)
            # NOTE: at the i-th day, we can see factor_exp[i] and stock_cap[i]
            # but cannot see stock_resid[i], so here use stock_resid[i-1]

        if adj_bayes:
            vola = adj_resid_vola_bayes(vola, stock_cap.iloc[i])

        if adj_vra:
            vola, lamb = adj_resid_vola_vra(vola, bias_stats[-max_T:], tau_vra)
            multipliers.append(lamb)
        else:
            multipliers.append(1.0)

        # update bias
        bias = stock_resid.iloc[i] / vola
        w = stock_cap.iloc[i].reindex(bias.index).values
        B = weighted_mean(bias.values**2, w) # NOTE: B^2
        bias_stats.append(B)

        res[date] = vola

    resid_vola = pd.DataFrame(res).T
    resid_vola = resid_vola.where(~stock_cap.iloc[min_T:].isna())

    bias_stats = pd.Series(bias_stats, index=stock_resid.index[min_T:])
    multipliers = pd.Series(multipliers, index=stock_resid.index[min_T:])

    gammas = pd.DataFrame(gammas).T

    return resid_vola, bias_stats, multipliers, gammas


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--outdir', default='data')
    parser.add_argument('--lags_nw', type=int, default=0) # disable nw
    parser.add_argument('--tau_nw', type=int, default=126) # disabled by lags_nw
    parser.add_argument('--tau_vola', type=int, default=60)
    parser.add_argument('--tau_vra', type=int, default=20)
    parser.add_argument('--max_T', type=int, default=480)
    parser.add_argument('--gamma_T', type=int, default=180) # structured risk
    parser.add_argument('--adj_str', action='store_true')
    parser.add_argument('--adj_bayes', action='store_true')
    parser.add_argument('--adj_vra', action='store_true', default=True)
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--clip_weight', action='store_false')

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()

    stock_cap = pd.read_pickle('data/cap.pkl').unstack().loc[pd.Timestamp('2015-01-01'):]
    stock_resid = pd.read_pickle(args.outdir + '/stock_resid.pkl').loc[pd.Timestamp('2015-01-01'):]
    factor_exp = pd.read_pickle('data/base.pkl').loc[pd.Timestamp('2015-01-01'):]

    if os.path.exists(args.outdir + '/pred.pkl'):
        df = pd.read_pickle(args.outdir + '/pred.pkl').loc[pd.Timestamp('2015-01-01'):]
        df.columns = ['RISK%d'%d for d in range(df.shape[1])]
        if args.replace:
            factor_exp = factor_exp.iloc[:, :-10]  # replace factors
        factor_exp[df.columns] = df

    print(factor_exp.head())

    resid_vola, bias_stats, multipliers, gammas = run(
        stock_resid, stock_cap, factor_exp,
        tau_vola=args.tau_vola, tau_nw=args.tau_nw,
        lags_nw=args.lags_nw, tau_vra=args.tau_vra,
        gamma_T=args.gamma_T, max_T=args.max_T,
        adj_str=args.adj_str, adj_bayes=args.adj_bayes,
        adj_vra=args.adj_vra, clip_weight=args.clip_weight)

    resid_vola.to_pickle(args.outdir + '/resid_vola.pkl')
    bias_stats.to_pickle(args.outdir + '/bias_stats_resvol.pkl')
    multipliers.to_pickle(args.outdir + '/multipliers_resvol.pkl')
    gammas.to_pickle(args.outdir + '/gammas.pkl')
