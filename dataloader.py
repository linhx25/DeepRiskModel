import torch
import numpy as np


class DataLoader:

    def __init__(self, df_feature, df_ret, df_cap, pin_memory=True, device=None):

        assert len(df_feature) == len(df_ret) and len(df_feature) == len(df_cap)

        self.df_feature = df_feature.values
        self.df_ret = df_ret.values
        self.df_cap = df_cap.values
        self.index = df_ret.index

        self.pin_memory = pin_memory
        self.device = device

        # pin as tensor
        if self.pin_memory:
            self.df_feature = torch.tensor(self.df_feature, dtype=torch.float, device=self.device)
            self.df_ret = torch.tensor(self.df_ret, dtype=torch.float, device=self.device)
            self.df_cap = torch.tensor(self.df_cap, dtype=torch.float, device=self.device)

        # build index
        self.daily_count = df_ret.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

        self.shuffle = False

    def train(self):
        self.shuffle = True

    def eval(self):
        self.shuffle = False

    def __len__(self):
        return len(self.daily_count)

    def __iter__(self):
        indices = np.arange(len(self.daily_count))
        first_index = 0
        if self.shuffle:
            # np.random.shuffle(indices)
            first_index = np.random.choice(32)  # only shuffle the first batch for prediction smoothing
        for idx in indices:
            if idx < first_index:
                continue
            slc = slice(self.daily_index[idx], self.daily_index[idx] + self.daily_count[idx])
            outs = self.df_feature[slc], self.df_ret[slc], self.df_cap[slc]
            if not self.pin_memory:
                outs = tuple(torch.tensor(x, dtype=torch.float, device=self.device) for x in outs)
            yield outs + (self.index[slc],)


if __name__ == '__main__':

    pass
