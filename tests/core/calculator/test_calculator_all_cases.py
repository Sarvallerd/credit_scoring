import pytest

from src.app.core.calculator import Calculator
from src.app.core.api import Features


class TestCalculator(object):

    @pytest.mark.parametrize(
        "proba, age, weighted_ext_scores, expected_amount",
        [
            # 150
            (0.01, 35, 0.1, 150),
            (0.01, 25, 0.1, 150),
            (0.08, 30, 0.2, 150),
            (0.2, 35, 0.6, 150),
            # 300
            (0.01, 25, 0.7, 300),
            (0.08, 25, 0.7, 300),
            (0.3, 25, 0.7, 300),
            (0.01, 35, 0.3, 300),
            (0.08, 34, 0.3, 300),
            (0.3, 35, 0.3, 300),
            (0.01, 26, 0.3, 300),
            (0.08, 27, 0.3, 300),
            (0.3, 27, 0.3, 300),
            (0.08, 35, 0.7, 300),
            # 500
            (0.014, 35, 0.6, 500),
        ],
    )
    def test_calc_amount(
            self,
            proba,
            age,
            weighted_ext_scores,
            expected_amount,
    ):
        calculator = Calculator()
        features = Features(
            weighted_ext_scores=weighted_ext_scores,
            age=age,
        )
        result = calculator.calc_amount(
            proba,
            features,
        )
        assert result == expected_amount
