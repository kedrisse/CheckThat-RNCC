

class ScoreNormalization:

    def __init__(self, sql_manager):
        self.sql_manager = sql_manager
        self.min_max_scores = self.sql_manager.get_min_max_scores()

    def get_normalize_score(self, criterion, score):
        """

        :param criterion: string, the name of the criterion
        :param score: float, the score to be normelized
        :return:
        """
        return round(normalize(self.min_max_scores, criterion, score), 2)


def normalize(min_max_scores, criterion, score):
    """
    Given a criterion and an original score, return then normalized score
    :param params: params array with min max infos for each criterias
    :param criterion: name of the criterion
    :param score: original score
    :return: the normalized score
    """
    min = min_max_scores[criterion]['min']
    max = min_max_scores[criterion]['max']

    max = max - min
    if max == 0.:
        return score
    score = score - min
    score = score * 100 / max
    return score
