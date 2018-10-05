from nltk import word_tokenize
from stop_words import get_stop_words


def get_technical_tokens():
    technical_tokens = []
    filename = "criterias/technical_words.txt"

    f = open(filename, "r", encoding="utf8")
    for line in f:
        technical_tokens.append(line[:-1])
    f.close()

    return technical_tokens


class Technicality:

    def __init__(self):
        self.technical_items = get_technical_tokens()
        self.text = ""
        self.tokens = []

    def find_technical_tokens(self):

        tech_items = []
        # for each word in text
        for i in range(len(self.tokens)):
            # if the word is in the technical list
            if self.is_controversial(self.tokens[i]):
                tech_items.append(self.tokens[i])

        return tech_items

    def is_controversial(self, token):
        for item in self.technical_items:
            if str.lower(item) == str.lower(token):
                return True

        return False

    def clean_tokens(self):
        to_be_remove = [',', '“', '”', '’', '.', ':', '—', '–', '-', '|', '[', ']', '(', ')', '$', '!', '‘']
        stop_words = get_stop_words('en')
        new_list = []
        for i in self.tokens:
            if i not in to_be_remove and str.lower(i) not in stop_words:
                new_list.append(i)
        self.tokens = new_list

    def score(self, text):
        self.text = text
        self.tokens = word_tokenize(self.text)
        self.clean_tokens()
        technical_tokens = self.find_technical_tokens()
        return len(technical_tokens) / len(self.tokens) if len(self.tokens) > 0 else 1