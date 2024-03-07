import qlib
# qlib.init(provider_uri='~/.qlib/qlib_data/cn_data')
qlib.init_from_yaml_conf('/data/csdesign/qlib_client_config/ycz_daily_offline.yaml')
from qlib.data import D
import pandas as pd

fields = []
names = []
windows = [240, 200, 180, 160, 140, 120, 100, 90, 80, 70, 60, 50, 45, 40, 35, 30, 25, 20, 15, 10] # 20
for field in ['open', 'high', 'low', 'close', 'volume']:
    for d in windows:
        names.append(field+str(d))
        if field != 'volume':
            fields.append(f"Log(Ref(${field}, {d}) / Mean($close, 60))")
        else:
            fields.append(f"Log(Ref(${field}, {d}) / Mean($volume, 60))")

print('load data')
df = D.features(D.instruments('all'), fields, start_time='2009-04-30', end_time='2020-02-10')
df.columns = names
df.index = df.index.swaplevel()
df.sort_index(inplace=True)

print('reindex data')
df = df.reindex(pd.read_pickle('ret.pkl').index)

print('normalize')
med = df.loc[:pd.Timestamp('2014-12-31')].median()
df -= med
mad = df.loc[:pd.Timestamp('2014-12-31')].abs().median()
mad *= 1.4826
mad.replace(0, 1, inplace=True)
df /= mad
df.clip(-3, 3, inplace=True)
df.fillna(0, inplace=True)

print('save')
pd.DataFrame({'mean': med, 'std': mad}).to_pickle('feature_stats.pkl')  # for only inference
df.to_pickle('feature.pkl')

label = D.features(D.instruments('all'),
                   ['Ref($close,-1)/$close-1'] + [f'Ref(Ref($close,-1)/$close-1,-%d)'%d for d in range(1,20)],
                   start_time='2009-04-30', end_time='2020-02-10')
label.index = label.index.swaplevel()
label.sort_index(inplace=True)
label.columns = ['R%d'%d for d in range(1, 21)]
label.reindex(pd.read_pickle('ret.pkl').index).to_pickle('label.pkl')
