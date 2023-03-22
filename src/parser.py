import pandas as pd
import json
import time

from dataclasses import dataclass
from dataclasses import asdict

DATA_PATH = "POS_CASH_balance_plus_bureau-001.log"
OUTPUT_BUREAU_PATH = 'parsed_bureau.csv'
OUTPUT_POS_CASH_BALANCE_PATH = 'parsed_POS.csv'


@dataclass
class AmtCredit:
    CREDIT_CURRENCY: str
    AMT_CREDIT_MAX_OVERDUE: float
    AMT_CREDIT_SUM: float
    AMT_CREDIT_SUM_DEBT: float
    AMT_CREDIT_SUM_LIMIT: float
    AMT_CREDIT_SUM_OVERDUE: float
    AMT_ANNUITY: float


@dataclass
class PosCashBalanceIDs:
    SK_ID_PREV: int
    SK_ID_CURR: int
    NAME_CONTRACT_STATUS: str


bureau_schema = []
POS_CASH_balance_schema = []


def parse_bureau(data: dict) -> None:
    row = dict()
    amt_credit_dict = asdict(eval(data['record']['AmtCredit']))

    row['SK_ID_CURR'] = data['record']['SK_ID_CURR']
    row['SK_ID_BUREAU'] = data['record']['SK_ID_BUREAU']
    row['CREDIT_ACTIVE'] = data['record']['CREDIT_ACTIVE']
    row['CREDIT_CURRENCY'] = amt_credit_dict['CREDIT_CURRENCY']
    row['DAYS_CREDIT'] = data['record']['DAYS_CREDIT']
    row['CREDIT_DAY_OVERDUE'] = data['record']['CREDIT_DAY_OVERDUE']
    row['DAYS_CREDIT_ENDDATE'] = data['record']['DAYS_CREDIT_ENDDATE']
    row['DAYS_ENDDATE_FACT'] = data['record']['DAYS_ENDDATE_FACT']
    row['AMT_CREDIT_MAX_OVERDUE'] = amt_credit_dict['AMT_CREDIT_MAX_OVERDUE']
    row['CNT_CREDIT_PROLONG'] = data['record']['CNT_CREDIT_PROLONG']
    row['AMT_CREDIT_SUM'] = amt_credit_dict['AMT_CREDIT_SUM']
    row['AMT_CREDIT_SUM_DEBT'] = amt_credit_dict['AMT_CREDIT_SUM_DEBT']
    row['AMT_CREDIT_SUM_LIMIT'] = amt_credit_dict['AMT_CREDIT_SUM_LIMIT']
    row['AMT_CREDIT_SUM_OVERDUE'] = amt_credit_dict['AMT_CREDIT_SUM_OVERDUE']
    row['CREDIT_TYPE'] = data['CREDIT_TYPE']
    row['DAYS_CREDIT_UPDATE'] = data['record']['DAYS_CREDIT_UPDATE']
    row['AMT_ANNUITY'] = amt_credit_dict['AMT_ANNUITY']

    bureau_schema.append(row)


def parse_pos_cash(data: dict) -> None:
    for record in data['records']:
        row = dict()
        pos_cash_balance_ids_dict = asdict(eval(record['PosCashBalanceIDs']))

        row['SK_ID_PREV'] = pos_cash_balance_ids_dict['SK_ID_PREV']
        row['SK_ID_CURR'] = pos_cash_balance_ids_dict['SK_ID_CURR']
        row['MONTHS_BALANCE'] = record['MONTHS_BALANCE']
        row['CNT_INSTALMENT'] = data['CNT_INSTALMENT']
        row['CNT_INSTALMENT_FUTURE'] = record['CNT_INSTALMENT_FUTURE']
        row['NAME_CONTRACT_STATUS'] = pos_cash_balance_ids_dict['NAME_CONTRACT_STATUS']
        row['SK_DPD'] = record['SK_DPD']
        row['SK_DPD_DEF'] = record['SK_DPD_DEF']

        POS_CASH_balance_schema.append(row)


def main(path: str, output_bureau_path: str, output_pos_cash_path: str) -> None:
    with open(path, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            data = json_line['data']
            if json_line['type'] == 'bureau':
                parse_bureau(data)
            else:
                parse_pos_cash(data)

    pd.DataFrame(bureau_schema).to_csv(output_bureau_path, index=False)
    pd.DataFrame(POS_CASH_balance_schema).to_csv(output_pos_cash_path, index=False)


start = time.time()
main(DATA_PATH, OUTPUT_BUREAU_PATH, OUTPUT_POS_CASH_BALANCE_PATH)
print(f'Скрипт выполнился за: {time.time() - start} сек')