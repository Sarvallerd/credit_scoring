import os
import sys
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0], '../../utils/'))
sys.path.insert(1, os.path.join(sys.path[0], '../../../config/'))

from db_connection import DB
from config import db_config

db = DB(db_config)

installments_payments = db.get_df_from_query("""SELECT * FROM installments_payments""")

installments_payments['flag_overdue'] = (installments_payments['days_entry_payment'] -
                                         installments_payments['days_instalment'])\
    .apply(lambda x: 0 if x < 0 else 1)
installments_payments['flag_overdue_amt'] = (installments_payments['amt_payment'] -
                                             installments_payments['amt_instalment'])\
    .apply(lambda x: 0 if x >= 0 else 1)

num_feature = ['sk_id_prev', 'sk_id_curr', 'num_instalment_number']
days_feature = ['sk_id_prev', 'sk_id_curr', 'flag_overdue']
amt_feature = ['sk_id_prev', 'sk_id_curr', 'flag_overdue_amt']

df_num_inst = installments_payments[num_feature].groupby(['sk_id_prev', 'sk_id_curr']).max().reset_index()
df_days_flag = installments_payments[days_feature].groupby(['sk_id_prev', 'sk_id_curr']).sum().reset_index(drop=True)
df_amt_flag = installments_payments[amt_feature].groupby(['sk_id_prev', 'sk_id_curr']).sum().reset_index(drop=True)

features = pd.concat([df_num_inst, df_days_flag, df_amt_flag], axis=1)

features.set_index('sk_id_prev', inplace=True)
features.to_csv('features_csv/installments_payments_features.csv')