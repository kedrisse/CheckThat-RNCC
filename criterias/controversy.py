import re
import ssl
import unicodedata
from difflib import SequenceMatcher

from nltk import pos_tag, word_tokenize
from stop_words import get_stop_words


def get_controversial_tokens():
    controversial_tokens = []
    filename = "criterias/controversial_topics.txt"

    f = open(filename, "r", encoding="utf8")
    for line in f:
        controversial_tokens.append(line[:-1])
    f.close()

    return controversial_tokens


def get_proper_nouns(tokens):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    tagged_tokens = pos_tag(tokens)

    res = []
    local_word = ''
    remove = 0
    tokens_to_remove = []
    for i in range(len(tagged_tokens)):
        word = tagged_tokens[i]
        prev = tagged_tokens[i - 1]
        next = tagged_tokens[i + 1] if i < len(tagged_tokens) - 1 else ('', None)

        if word[1] == 'NNP' and (prev[1] == 'NNP' or next[1] == 'NNP'):
            local_word += ' ' + word[0]
            remove += 1
        else:
            if local_word != '':
                res.append(local_word.lstrip())
                for j in range(remove,0,-1):
                    tokens_to_remove.append(i-j)
            local_word = ''
            remove = 0

    for index in tokens_to_remove[::-1]:
        del tokens[index]

    return res, tokens


class Controversy:

    def __init__(self):
        self.controversial_items = get_controversial_tokens()
        self.text = ""
        self.tokens = []

    def build_tokens(self):
        tokens = word_tokenize(self.text)
        proper_nouns, toks = get_proper_nouns(tokens)
        self.tokens = toks + proper_nouns
        self.clean_tokens()

    def find_controversial_tokens(self):

        contro_items = []
        # for each word in text
        for i in range(len(self.tokens)):
            # if the word is in the controversial list
            if self.is_controversial(self.tokens[i]):
                contro_items.append(self.tokens[i])

        return contro_items

    def is_controversial(self, token):
        for item in self.controversial_items:
            # if (similar(item, token)) > 0.9:
            # if parameterize(item) == parameterize(token):
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
        self.build_tokens()
        controversial_tokens = self.find_controversial_tokens()
        return len(controversial_tokens) / len(self.tokens) if len(self.tokens) > 0 else 1
