from src.app.core.api import Features


class Calculator(object):
    """Класс для расчета суммы выдачи."""

    _age = 30
    _ext_scores150 = 0.2
    _ext_scores300 = 0.4
    _threshold500 = 0.056
    _threshold300 = 0.09

    _amount = 500
    _amount300 = 300
    _amount150 = 150

    def calc_amount(
        self,
        proba: float,
        features: Features,
    ) -> int:
        """Рассчет суммы кредита для выдачи."""
        age = features.age
        ext_scores = features.weighted_ext_scores

        if ext_scores <= self._ext_scores150:
            return self._amount150

        if age <= self._age or ext_scores <= self._ext_scores300:
            return self._amount300

        if proba < self._threshold500:
            return self._amount

        if proba < self._threshold300:
            return self._amount300

        return self._amount150
