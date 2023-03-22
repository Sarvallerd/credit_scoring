from feature_utils import features_selection

DATA_PATH = "../features/features_csv/previous_application_features.csv "
OUTPUT_PATH = "mfeatures_csv/previous_application_mfeatures.csv"


def main(DATA_PATH: str, OUTPUT_PATH: str) -> None:
    features_selection(DATA_PATH, OUTPUT_PATH)


main(DATA_PATH, OUTPUT_PATH)