import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config

db = DB(db_config)

previous_application = db.get_df_from_query("""SELECT * FROM previous_application""")

amt_features = ['sk_id_curr', 'amt_annuity', 'amt_application', 'amt_credit', 'amt_down_payment', 'amt_goods_price']
df_all_agg_amt = previous_application[amt_features]\
    .groupby('sk_id_curr')\
    .agg(['max', 'min', 'mean', 'median', 'sum', 'std'])
df_all_agg_amt.columns = [multidx[0] + '_' + multidx[1] for multidx in df_all_agg_amt.columns]
df_all_agg_amt = df_all_agg_amt.reset_index()

flag_features = ['sk_id_curr', 'flag_last_appl_per_contract', 'nflag_last_appl_in_day', 'nflag_insured_on_approval']
previous_application['flag_last_appl_per_contract'] = previous_application['flag_last_appl_per_contract'].map({
    'Y': 1,
    'N': 0
})
df_all_agg_flags = previous_application[flag_features]\
    .groupby('sk_id_curr')\
    .agg(flag_last_appl_per_contract_sum=pd.NamedAgg(column='flag_last_appl_per_contract', aggfunc='sum'),
         nflag_last_appl_in_day_sum=pd.NamedAgg(column='nflag_last_appl_in_day', aggfunc='sum'),
         nflag_insured_on_approval_sum=pd.NamedAgg(column='nflag_insured_on_approval', aggfunc='sum'))\
    .reset_index(drop=True)

rate_features = ['sk_id_curr', 'rate_down_payment', 'rate_interest_primary', 'rate_interest_privileged']
df_all_agg_rate = previous_application[rate_features].groupby('sk_id_curr').agg(['mean', 'median'])
df_all_agg_rate.columns = [multidx[0] + '_' + multidx[1] for multidx in df_all_agg_rate.columns]
df_all_agg_rate = df_all_agg_rate.reset_index(drop=True)

days_features = ['sk_id_curr', 'days_first_drawing', 'days_first_due', 'days_last_due']
df_all_agg_days = previous_application[days_features].groupby('sk_id_curr').agg(['min', 'max'])
df_all_agg_days.columns = [multidx[0] + '_' + multidx[1] for multidx in df_all_agg_days.columns]
df_all_agg_days = df_all_agg_days.reset_index(drop=True)

features = pd.concat([df_all_agg_amt, df_all_agg_flags, df_all_agg_rate, df_all_agg_days], axis=1)

features.set_index('sk_id_curr', inplace=True)
features.to_csv('features_csv/previous_application_features.csv')