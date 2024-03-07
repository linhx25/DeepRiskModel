import os
import copy
import json
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from loguru import logger

from model import Model
from dataloader import DataLoader
from function import regression, zscore


global_step = -1
def train_epoch(epoch, model, optimizer, train_loader, writer, args):

    global global_step

    model.train()
    train_loader.train()
    optimizer.zero_grad()  # always reset grad

    tmp_r2 = 0
    tmp_norm = 0
    total_r2 = 0
    total_norm = 0
    prev_pred = None
    prev_index = None
    for step, (feature, ret, cap, index) in enumerate(tqdm(train_loader)):
        global_step += 1

        if len(ret.shape) == 2 and args.next_ret_only:
            ret = ret[:, 0]

        pred = model(feature)

        # smoothing
        cur_index = index.get_level_values(0)
        if prev_pred is not None:
            shared_index = prev_index.intersection(cur_index)
            cur_mask = cur_index.isin(shared_index, level=0)
            prev_mask = prev_index.isin(shared_index, level=0)
            pred[cur_mask] = prev_pred[prev_mask] * args.rho + pred[cur_mask] * (1 - args.rho)
        prev_pred = pred.detach()
        prev_index = cur_index

        _, _, r2, norm = regression(pred, ret, cap)

        loss = - r2 + args.lamb * norm
        loss /= args.update_freq
        loss.backward()

        tmp_r2 += r2.item()
        tmp_norm += norm.item()

        if (step + 1) % args.update_freq == 0:

            optimizer.step()
            optimizer.zero_grad()

            total_r2 += tmp_r2
            total_norm += tmp_norm

            if writer is not None:
                tmp_r2 /= args.update_freq
                tmp_norm /= args.update_freq
                writer.add_scalar('train/r2', tmp_r2, global_step // args.update_freq)
                writer.add_scalar('train/norm', tmp_norm, global_step // args.update_freq)
                writer.add_scalar('train/loss', - tmp_r2 + args.lamb * tmp_norm, global_step // args.update_freq)

            tmp_r2 = 0
            tmp_norm = 0

        torch.cuda.empty_cache() # docker memory leak

    total_r2 /= len(train_loader)
    total_norm /= len(train_loader)

    return total_r2, total_norm, - total_r2 + args.lamb * total_norm


def test_epoch(epoch, model, test_loader, writer, args, prefix='test', save_pred=False):

    model.eval()
    test_loader.eval()

    total_r2 = 0
    total_norm = 0
    preds = []
    prev_pred = None
    prev_index = None
    for feature, ret, cap, index in tqdm(test_loader):

        if len(ret.shape) == 2:
            ret = ret[:, 0]

        with torch.no_grad():
            pred = model(feature)

        # smoothing
        cur_index = index.get_level_values(0)
        if prev_pred is not None:
            shared_index = prev_index.intersection(cur_index)
            cur_mask = cur_index.isin(shared_index, level=0)
            prev_mask = prev_index.isin(shared_index, level=0)
            pred[cur_mask] = prev_pred[prev_mask] * args.rho + pred[cur_mask] * (1 - args.rho)
        prev_pred = pred
        prev_index = cur_index

        _, _, r2, norm = regression(pred, ret, cap)
        total_r2 += r2.item()
        total_norm += norm.item()

        if save_pred:
            pred = zscore(pred, cap, mask_w=True)  # cap has nan
            preds.append(pd.DataFrame(pred.cpu().numpy(), index=index))

        torch.cuda.empty_cache() # docker memory leak

    total_r2 /= len(test_loader)
    total_norm /= len(test_loader)
    total_loss = - total_r2 + args.lamb * total_norm

    if writer is not None:
        writer.add_scalar(prefix+'/r2', total_r2, epoch)
        writer.add_scalar(prefix+'/norm', total_norm, epoch)
        writer.add_scalar(prefix+'/loss', total_loss, epoch)

    if len(preds):
        preds = pd.concat(preds, axis=0)

    return total_r2, total_norm, total_loss, preds


def create_loaders(args):

    logger.info('load data')
    df_feature = pd.read_pickle(args.datadir + '/' + args.feature + '.pkl')
    df_ret = pd.read_pickle(args.datadir + '/' + args.ret + '.pkl')
    try:
        df_cap = pd.read_pickle(args.datadir + '/' +'cap.pkl')
    except:
        logger.warning('market cap is not found, will use identity weight.')
        df_cap = pd.Series(1, index=df_ret.index, dtype='float32')

    assert len(df_feature) == len(df_ret) == len(df_cap)

    logger.info('init loader')
    slc = slice(args.train_start_date, args.train_end_date)
    train_loader = DataLoader(df_feature.loc[slc], df_ret.loc[slc], df_cap.loc[slc],
                              pin_memory=args.pin_memory, device=args.device)

    slc = slice(args.valid_start_date, args.valid_end_date)
    valid_loader = DataLoader(df_feature.loc[slc], df_ret.loc[slc], df_cap.loc[slc],
                              pin_memory=args.pin_memory, device=args.device)

    slc = slice(args.test_start_date, args.test_end_date)
    test_loader = DataLoader(df_feature.loc[slc], df_ret.loc[slc], df_cap.loc[slc],
                              pin_memory=args.pin_memory, device=args.device)

    return train_loader, valid_loader, test_loader


def main(args):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.outdir, exist_ok=True)

    if not args.overwrite and os.path.exists(args.outdir+'/'+'info.json'):
        print('already finished, exit.')
        return

    writer = SummaryWriter(log_dir=args.outdir) if args.n_epochs > 0 else None

    logger.info('create model...')
    model = Model(**vars(args)).to(args.device)
    if args.init_state:
        logger.info('load model from init state')
        model.load_state_dict(torch.load(args.init_state, map_location='cpu'))
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    logger.info('# params: %d' % sum([p.numel() for p in model.parameters()]))

    logger.info('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    best_score = -1
    best_epoch = 0
    stop_round = 0
    best_param = copy.deepcopy(model.state_dict())
    for epoch in range(args.n_epochs):
        logger.info('Epoch: %d' % epoch)

        logger.info('training...')
        train_r2, train_norm, train_loss = train_epoch(epoch, model, optimizer, train_loader, writer, args)

        logger.info('evaluating...')
        valid_r2, valid_norm, valid_loss, _ = test_epoch(epoch, model, valid_loader, writer, args, prefix='valid')
        test_r2, test_norm, test_loss, _ = test_epoch(epoch, model, test_loader, writer, args, prefix='test')

        logger.info('r2 - train %.6f, valid %.6f, test %.6f'%(train_r2, valid_r2, test_r2))
        logger.info('norm - train %.6f, valid %.6f, test %.6f'%(train_norm, valid_norm, test_norm))

        if valid_r2 > best_score:
            logger.info(f'\tvalid r2 update from {best_score:.6f} to {valid_r2:.6f}')
            best_score = valid_r2
            stop_round = 0
            best_epoch = epoch
            best_param = copy.deepcopy(model.state_dict())
            torch.save(best_param, args.outdir+'/model.bin')
        else:
            stop_round += 1
            if stop_round >= args.early_stop:
                logger.info('early stop')
                break

    logger.info(f'best r2: {best_score:.6f} @ {best_epoch}')
    model.load_state_dict(best_param)

    logger.info('inference...')
    pred = []
    for name in ['train', 'valid', 'test']:
        pred.append(test_epoch(-1, model, eval(name+'_loader'), None, args, prefix=name, save_pred=True)[-1])
    pd.concat(pred, axis=0).to_pickle(args.outdir+'/pred.pkl')

    logger.info('save info...')
    info = dict(
        config=vars(args),
        best_epoch=best_epoch,
        best_score=best_score,
    )
    default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
    with open(args.outdir+'/info.json', 'w') as f:
        json.dump(info, f, default=default, indent=4)

    logger.info('finished.')


class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                # FIXME: hard code date convert
                if 'date' in key:
                    value = pd.Timestamp(value)
                setattr(namespace, key, value)


def parse_args():
    parser = argparse.ArgumentParser()
    # model
    parser.add_argument('--model_name', default='GRU')
    parser.add_argument('--input_size', type=int, default=5)
    parser.add_argument('--hidden_size', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_factors', type=int, default=10)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropatt', type=float, default=0.5)
    parser.add_argument('--disable_gat', action='store_true')
    # training
    parser.add_argument('--rho', type=float, default=0.99)
    parser.add_argument('--lamb', type=float, default=0.01)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--update_freq', type=int, default=64)
    parser.add_argument('--next_ret_only', action='store_true')
    parser.add_argument('--add_intercept', action='store_true', default=True)  # default=True -> disable this argument
    parser.add_argument('--clip_weight', action='store_true', default=True)  # default=True -> disable this argument
    # data
    parser.add_argument('--pin_memory', action='store_true', default=True)  # default=True -> disable this argument
    parser.add_argument('--train_start_date', type=pd.Timestamp, default='2007-01-01')
    parser.add_argument('--train_end_date', type=pd.Timestamp, default='2014-12-31')
    parser.add_argument('--valid_start_date', type=pd.Timestamp, default='2015-01-01')
    parser.add_argument('--valid_end_date', type=pd.Timestamp, default='2016-12-31')
    parser.add_argument('--test_start_date', type=pd.Timestamp, default='2017-01-01')
    parser.add_argument('--test_end_date', type=pd.Timestamp, default='2099-12-31')
    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--config', action=ParseConfigFile)
    parser.add_argument('--init_state', default=None)
    parser.add_argument('--datadir', default='./data')
    parser.add_argument('--feature', default='feature')
    parser.add_argument('--ret', default='label')
    parser.add_argument('--outdir', default='./output')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = parse_args()

    main(args)
