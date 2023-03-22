import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config

db = DB(db_config)

credit_card_balance = db.get_df_from_query("""SELECT * FROM credit_card_balance""")

unique_ids = pd.DataFrame(data=sorted(credit_card_balance['sk_id_curr'].unique()), columns=['sk_id_curr'])
unique_ids = unique_ids.merge(credit_card_balance[['sk_id_curr', 'sk_id_prev']], how='left', on='sk_id_curr')\
    .drop_duplicates()\
    .reset_index(drop=True)

df_all_agg = credit_card_balance.drop(columns=['months_balance', 'name_contract_status'])\
    .groupby(['sk_id_curr', 'sk_id_prev'])\
    .agg(['max', 'min', 'mean', 'median', 'sum', 'var', 'std'])
df_all_agg.columns = [multidx[0] + '_' + multidx[1] for multidx in df_all_agg.columns]
df_all_agg = df_all_agg.reset_index()

df_3_month_agg = credit_card_balance.loc[credit_card_balance['months_balance'].isin((-1, -2, -3))]
df_3_month_agg = df_3_month_agg.drop(columns=['months_balance', 'name_contract_status'])\
    .groupby(['sk_id_curr', 'sk_id_prev'])\
    .agg(['max', 'min', 'mean', 'median', 'sum', 'var', 'std'])
df_3_month_agg.columns = [multidx[0] + '_' + multidx[1] for multidx in df_3_month_agg.columns]
df_3_month_agg = df_3_month_agg.merge(unique_ids, on=['sk_id_curr', 'sk_id_prev'], how='right')

df_diff = (df_all_agg - df_3_month_agg).drop(columns=['sk_id_curr', 'sk_id_prev'])
df_diff.columns = [f'{col}_diff_3m' for col in df_diff.columns]

features = pd.concat([df_all_agg, df_diff], axis=1)

features.set_index('sk_id_prev', inplace=True)
features.to_csv('features_csv/credit_card_features.csv')