import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config

db = DB(db_config)

pos_cash_balance = db.get_df_from_query("""SELECT * FROM pos_cash_balance""")

df_all_agg = pos_cash_balance.groupby(['sk_id_curr', 'sk_id_prev'])['cnt_instalment_future', 'cnt_instalment',
                                                                    'sk_dpd', 'sk_dpd_def']\
    .agg(['mean', 'sum'])
df_all_agg.columns = [multidx[0] + '_' + multidx[1] + '_all' for multidx in df_all_agg.columns]
df_all_agg = df_all_agg.reset_index()

df_3years_agg = pos_cash_balance.loc[pos_cash_balance['months_balance'] >= -36].groupby(['sk_id_curr', 'sk_id_prev'])['cnt_instalment_future', 'cnt_instalment',
                                                                                                                      'sk_dpd', 'sk_dpd_def']\
    .agg(['mean', 'sum'])
df_3years_agg.columns = [multidx[0] + '_' + multidx[1] + '3_years' for multidx in df_3years_agg.columns]
df_3years_agg = df_3years_agg.reset_index()

features = df_all_agg.merge(df_3years_agg.drop(columns='sk_id_curr'), on='sk_id_prev', how='left')

features.set_index('sk_id_prev', inplace=True)
features.to_csv('features_csv/pos_cash_balance.csv')