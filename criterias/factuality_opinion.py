import pickle

import numpy as np


def divide_into_sentences(document):
    return [sent for sent in document.sents]


def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()

    return file


def number_of_fine_grained_pos_tags(sent):
    """
    Find all the tags related to words in a given sentence. Slightly more
    informative then part of speech tags, but overall similar data.
    Only one might be necessary.
    For complete explanation of each tag, visit: https://spacy.io/api/annotation
    """
    tag_dict = {
        '-LRB-': 0, '-RRB-': 0, ',': 0, ':': 0, '.': 0, "''": 0, '""': 0, '#': 0,
        '``': 0, '$': 0, 'ADD': 0, 'AFX': 0, 'BES': 0, 'CC': 0, 'CD': 0, 'DT': 0,
        'EX': 0, 'FW': 0, 'GW': 0, 'HVS': 0, 'HYPH': 0, 'IN': 0, 'JJ': 0, 'JJR': 0,
        'JJS': 0, 'LS': 0, 'MD': 0, 'NFP': 0, 'NIL': 0, 'NN': 0, 'NNP': 0, 'NNPS': 0,
        'NNS': 0, 'PDT': 0, 'POS': 0, 'PRP': 0, 'PRP$': 0, 'RB': 0, 'RBR': 0, 'RBS': 0,
        'RP': 0, '_SP': 0, 'SYM': 0, 'TO': 0, 'UH': 0, 'VB': 0, 'VBD': 0, 'VBG': 0,
        'VBN': 0, 'VBP': 0, 'VBZ': 0, 'WDT': 0, 'WP': 0, 'WP$': 0, 'WRB': 0, 'XX': 0,
        'OOV': 0, 'TRAILING_SPACE': 0}

    for token in sent:
        if token.is_oov:
            tag_dict['OOV'] += 1
        elif token.tag_ == '':
            tag_dict['TRAILING_SPACE'] += 1
        else:
            tag_dict[token.tag_] += 1

    return tag_dict


def number_of_dependency_tags(sent):
    """
    Find a dependency tag for each token within a sentence and add their amount
    to a distionary, depending how many times that particular tag appears.
    """
    dep_dict = {
        'acl': 0, 'advcl': 0, 'advmod': 0, 'amod': 0, 'appos': 0, 'aux': 0, 'case': 0,
        'cc': 0, 'ccomp': 0, 'clf': 0, 'compound': 0, 'conj': 0, 'cop': 0, 'csubj': 0,
        'dep': 0, 'det': 0, 'discourse': 0, 'dislocated': 0, 'expl': 0, 'fixed': 0,
        'flat': 0, 'goeswith': 0, 'iobj': 0, 'list': 0, 'mark': 0, 'nmod': 0, 'nsubj': 0,
        'nummod': 0, 'obj': 0, 'obl': 0, 'orphan': 0, 'parataxis': 0, 'prep': 0, 'punct': 0,
        'pobj': 0, 'dobj': 0, 'attr': 0, 'relcl': 0, 'quantmod': 0, 'nsubjpass': 0,
        'reparandum': 0, 'ROOT': 0, 'vocative': 0, 'xcomp': 0, 'auxpass': 0, 'agent': 0,
        'poss': 0, 'pcomp': 0, 'npadvmod': 0, 'predet': 0, 'neg': 0, 'prt': 0, 'dative': 0,
        'oprd': 0, 'preconj': 0, 'acomp': 0, 'csubjpass': 0, 'meta': 0, 'intj': 0,
        'TRAILING_DEP': 0}

    for token in sent:
        if token.dep_ == '':
            dep_dict['TRAILING_DEP'] += 1
        else:
            try:
                dep_dict[token.dep_] += 1
            except:
                print('Unknown dependency for token: "' + token.orth_ + '". Passing.')

    return dep_dict


def number_of_specific_entities(sent):
    """
    Finds all the entities in the sentence and returns the amont of
    how many times each specific entity appear in the sentence.
    """
    entity_dict = {
        'PERSON': 0, 'NORP': 0, 'FAC': 0, 'ORG': 0, 'GPE': 0, 'LOC': 0,
        'PRODUCT': 0, 'EVENT': 0, 'WORK_OF_ART': 0, 'LAW': 0, 'LANGUAGE': 0,
        'DATE': 0, 'TIME': 0, 'PERCENT': 0, 'MONEY': 0, 'QUANTITY': 0,
        'ORDINAL': 0, 'CARDINAL': 0}

    entities = [ent.label_ for ent in sent.as_doc().ents]
    for entity in entities:
        entity_dict[entity] += 1

    return entity_dict


class FactualityOpinion:

    def __init__(self, nlp):
        self.classifier = load_pickle('criterias/models/nn_classifier.pickle')
        self.nlp = nlp

    def classify(self, article_text):
        fact_sents = 0.
        opinion_sents = 0.
        # Preprocess using spacy
        parsed_article = divide_into_sentences(self.nlp(article_text))
        nb_sents = len(parsed_article)

        for sentence in parsed_article:
            # Get features
            sentence_with_features = {}
            entities_dict = number_of_specific_entities(sentence)
            sentence_with_features.update(entities_dict)
            pos_dict = number_of_fine_grained_pos_tags(sentence)
            sentence_with_features.update(pos_dict)
            dep_dict = number_of_dependency_tags(sentence)
            sentence_with_features.update(dep_dict)

            # Transform features into array
            vals = np.fromiter(iter(sentence_with_features.values()), dtype=float)

            # Run a prediction
            # prediction = self.classifier.predict(vals.reshape(1, -1))
            # if prediction == 0:
            #    fact_sents += 1
            # else:
            #    opinion_sents += 1

            prediction = self.classifier.predict_proba(vals.reshape(1, -1))

        # return fact_sents/nb_sents if nb_sents > 0 else 0., opinion_sents/nb_sents if nb_sents > 0 else 0., fact_sents, opinion_sents, nb_sents
        return prediction[0][0]
