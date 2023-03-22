from models_utils import tuning


DATA_PATH = "../app/modelling/features/features_csv/"
files_names = ["application_features.csv", "bureau_balance_features.csv", "bureau_features.csv",
               "credit_card_features.csv", "installments_payments_features.csv", "pos_cash_balance.csv",
               "previous_application_features.csv"]

OUTPUT_LOGREG = "models/logreg.pkl"


def main(DATA_PATH: str, files_names: list, OUTPUT_LOGREG: str) -> None:
    """Вывод на печать скора лучшей модели логистической регрессии."""
    score = tuning(DATA_PATH, files_names, OUTPUT_LOGREG)
    print(f"ROC-AUC скор логистической регрессии: {score}")


main(DATA_PATH, files_names, OUTPUT_LOGREG)
