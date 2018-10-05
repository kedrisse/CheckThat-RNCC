import nltk
from criterias.AFINN import emotion_tab

class Emotion:

    def __init__(self, text):
        self.text = text

    def get_score(article):

        # version AFINN
        cpt_neg = 0
        cpt_pos = 0
        cpt_mots = 0

        tokens = nltk.word_tokenize(article)
        for elem in tokens:
            cpt_mots += 1
            if elem in emotion_tab:
                if emotion_tab[elem] < 0:
                    cpt_neg += float(emotion_tab[elem])
                else:
                    cpt_pos += float(emotion_tab[elem])

        if cpt_mots == 0:
            return 0

        cpt_pos = cpt_pos / cpt_mots
        cpt_neg = cpt_neg / cpt_mots

        # score = abs(cpt_neg) + cpt_pos

        #return score, (abs(cpt_neg)/score*100.), (cpt_pos/score*100.)
        return cpt_pos, cpt_neg
