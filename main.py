# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import spacy
from gensim import utils
from gensim.models import KeyedVectors
from gensim.models import word2vec
from sklearn import ensemble
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier

from criterias.controversy import Controversy
from criterias.emotion import Emotion
from criterias.factuality_opinion import FactualityOpinion
from criterias.technicality import Technicality

cwd = os.getcwd()


def float_to_str(f):
    """
    Convert the given float to a string,
    without resorting to scientific notation
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')


def parse(filename):
    f = open(filename, encoding="utf8")
    array = []

    for line in f:
        parts = line.split('\t')
        parts[0] = int(parts[0])
        parts[len(parts) - 1] = parts[len(parts) - 1].replace('\n', '')
        parts.append(0)
        array.append(parts)

    return array


def parse_labeled_file(filename):
    f = open(filename, encoding="utf8")
    array = []

    for line in f:
        parts = line.split('\t')
        parts[0] = int(parts[0])
        parts[len(parts) - 1] = parts[len(parts) - 1].replace('\n', '')
        parts[len(parts) - 1] = int(parts[len(parts) - 1])
        array.append(parts)

    return array


def calculer_score(text, controversy, technicality):
    controversy_score = controversy.score(text)
    fact_score = FactualityOpinion(nlp).classify(text)
    technicality_score = technicality.score(text)
    emotion_pos_score, emotion_neg_score = Emotion.get_score(text)
    # print(text+" calculé")
    return [controversy_score, fact_score, technicality_score, emotion_pos_score, emotion_neg_score]


def same_speaker(speaker1, speaker2):
    if speaker1 == "SYSTEM" or speaker2 == "SYSTEM":
        return 2
    elif speaker1 != speaker2:
        return 1

    return 0


def save_pickle(data, filepath):
    save_documents = open(filepath, 'wb')
    pickle.dump(data, save_documents)
    save_documents.close()


def load_pickle(filepath):
    documents_f = open(filepath, 'rb')
    file = pickle.load(documents_f)
    documents_f.close()

    return file


def normalize(predictions, score):
    max_ = max(predictions)
    min_ = min(predictions)

    max_ = max_ - min_
    if max_ == 0.:
        return score
    score = score - min_
    score = score / max_
    return score


def divide_into_sentences(document):
    return [sent for sent in document.sents]


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


def get_df(test_sent):
    # Preprocess using spacy
    parsed_test = divide_into_sentences(nlp(test_sent))
    if len(parsed_test) < 0:
        parsed_test.append('')
    # Get features
    sentence_with_features = {}
    entities_dict = number_of_specific_entities(parsed_test[0])
    sentence_with_features.update(entities_dict)
    pos_dict = number_of_fine_grained_pos_tags(parsed_test[0])
    sentence_with_features.update(pos_dict)
    dep_dict = number_of_dependency_tags(parsed_test[0])
    sentence_with_features.update(dep_dict)

    df = np.fromiter(iter(sentence_with_features.values()), dtype=float)

    return df.reshape(1, -1)


def sentence_features(sentence):
    features = []
    if use_label:
        features = [calculer_score(sentence, controversy, technicality)]
    if use_spacy:
        features = np.append(features, get_df(sentence)[0])
    if use_w2v:
        sentence_vector = []
        for word in utils.simple_preprocess(sentence):
            try:
                sentence_vector.append(word_vectors.wv[word])
            except KeyError:
                pass
        if len(sentence_vector) == 0:
            sentence_vector.append(np.zeros_like(word_vectors.wv["tax"]))
        features = np.append(features, np.mean(sentence_vector, axis=0))

    return features


def trainModel(X, y, method, oversample):
    kindSMOTE = 'regular'

    if method == 'random_forest':
        classifier = ensemble.RandomForestClassifier()
    elif method == 'svc_rbf':
        classifier = svm.SVC(probability=True)
        kindSMOTE = 'svm'
    elif method == 'sgd_log':
        classifier = SGDClassifier(loss='log')
        kindSMOTE = 'svm'
    elif method == 'nn_lbfgs':
        classifier = MLPClassifier(solver='lbfgs')
    else:
        classifier = svm.SVC(probability=True, kernel='linear')
        kindSMOTE = 'svm'

    if oversample:
        method += '_oversampled'
        # print("Training " + method)
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(kind=kindSMOTE)
        X_resampled, y_resampled = smote.fit_sample(X, y)
        classifier.fit(X_resampled, y_resampled)
        save_pickle(classifier, cwd + '/models/' + method + '_classifier.pickle')
    else:
        # print("Training " + method)
        classifier.fit(X, y)
        save_pickle(classifier, cwd + '/models/' + method + '_classifier.pickle')

    # if method == 'random_forest' or method == 'random_forest_oversampled':
    #     print("Coefs : " + str(classifier.feature_importances_))
    # elif method != 'nn_lbfgs' and method != 'nn_lbfgs_oversampled' and method != 'svc_rbf' and method != 'svc_rbf_oversampled':
    #     print("Coefs : " + str(classifier.coef_))


def trainSet(train_data):
    X = []
    y = []
    vectors = []

    if speakers:
        speakers_arr = []
        for i in train_data:
            speakers_arr.append(i[1])
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(train_data)):
            x_i = []

            for previous in range(surround_scope, 0, -1):
                if int(train_data[i][0]) - previous > 0:
                    x_i = np.append(np.append(x_i, vectors[i - previous]),
                                    [same_speaker(speakers_arr[i], speakers_arr[i - previous])])
                else:
                    x_i = np.append(np.append(x_i, np.zeros_like(vectors[i])), [2])

            x_i = np.append(np.append(x_i, vectors[i]), [0])

            for next in range(surround_scope):
                if i + next + 1 >= len(train_data) or train_data[i][0] + next + 1 > int(
                        train_data[i + 1][
                            0]):
                    x_i = np.append(np.append(x_i, np.zeros_like(vectors[i])), [2])
                else:
                    x_i = np.append(np.append(x_i, vectors[i + next + 1]),
                                    [same_speaker(speakers_arr[i], speakers_arr[i + next + 1])])

            X.append(x_i)
            y.append(train_data[i][3])
    else:
        for i in train_data:
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(train_data)):
            x_i = []

            for previous in range(surround_scope, 0, -1):
                if int(train_data[i][0]) - previous > 0:
                    x_i = np.append(x_i, vectors[i - previous])
                else:
                    x_i = np.append(x_i, np.zeros_like(vectors[i]))

            x_i = np.append(x_i, vectors[i])

            for next in range(surround_scope):
                if i + next + 1 >= len(train_data) or train_data[i][0] + next + 1 > int(
                        train_data[i + 1][
                            0]):
                    x_i = np.append(x_i, np.zeros_like(vectors[i]))
                else:
                    x_i = np.append(x_i, vectors[i + next + 1])

            X.append(x_i)
            y.append(train_data[i][3])

    return X, y


def testSet(data):
    to_predict = []
    vectors = []

    if speakers:
        speakers_arr = []
        for i in data:
            speakers_arr.append(i[1])
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        for i in range(len(data)):
            to_predict.append([])

            for previous in range(surround_scope, 0, -1):
                if i - previous >= 0:
                    to_predict[i] = np.append(np.append(to_predict[i], vectors[i - previous]),
                                              [same_speaker(speakers_arr[i], speakers_arr[i - previous])])
                else:
                    to_predict[i] = np.append(np.append(to_predict[i], np.zeros_like(vectors[i])), [2])

            to_predict[i] = np.append(np.append(to_predict[i], vectors[i]), [0])

            for next in range(surround_scope):
                if i + next + 1 < len(data):
                    to_predict[i] = np.append(np.append(to_predict[i], vectors[i + next + 1]),
                                              [same_speaker(speakers_arr[i], speakers_arr[i + next + 1])])
                else:
                    to_predict[i] = np.append(np.append(to_predict[i], np.zeros_like(vectors[i])), [2])
    else:
        for i in data:
            sentence = i[2]
            vectors.append(sentence_features(sentence))

        to_predict = []
        for i in range(len(data)):
            to_predict.append([])

            for previous in range(surround_scope, 0, -1):
                if i - previous >= 0:
                    to_predict[i] = np.append(to_predict[i], vectors[i - previous])
                else:
                    to_predict[i] = np.append(to_predict[i], np.zeros_like(vectors[i]))

            to_predict[i] = np.append(to_predict[i], vectors[i])

            for next in range(surround_scope):
                if i + next + 1 < len(data):
                    to_predict[i] = np.append(to_predict[i], vectors[i + next + 1])
                else:
                    to_predict[i] = np.append(to_predict[i], np.zeros_like(vectors[i]))

    return to_predict


if __name__ == '__main__':
    import decimal

    # Max number of digits for the computed scores
    ctx = decimal.Context()
    ctx.prec = 20

	# Number of sentences before and after the one to evaluate that we take in account
    surround_scope = 0
	# True if we take in account the name of the speaker
    speakers = False

	# Methods used
    use_label = False
    use_spacy = False
    use_w2v = True

	# True if we want to train models
    train = True
	# True if we want to test models
    test = True

	# True if we want to average the scores given by each model
    combine = True
	
	# [ML algorithm used, oversampling]
    All_models = [['random_forest', True], ['random_forest', False], ['svc_rbf', True], ['svc_rbf', False],
                  ['sgd_log', True], ['sgd_log', False], ['nn_lbfgs', True], ['nn_lbfgs', False], ['svc_linear', True],
                  ['svc_linear', False]]

    if train or test:
        if use_label:
            controversy = Controversy()
            technicality = Technicality()
        if use_label or use_spacy:
            model_size = 'md'
            nlp = spacy.load('en_core_web_' + model_size)
            # print(model_size + " model loaded")
        if use_w2v:
            word_vectors = KeyedVectors.load_word2vec_format(cwd + "/w2v models/GoogleNews-vectors-negative300.bin",
                                                             binary=True)
            # print("w2v model loaded")

    if train:
		# The train data is in "train_data.txt"
        train_data_array = parse_labeled_file(cwd + "/train_data.txt")
        X, y = trainSet(train_data_array)

        for model in All_models:
            trainModel(X, y, model[0], model[1])

	# Compute the scores of the sentences in each txt file in the folder "english"
    for filename in os.listdir('english'):
        if filename not in []:
            if test:
                data = parse(cwd + "/english/" + filename)
                to_predict = testSet(data)

                for model in All_models:
                    if model[1]:
                        classifier = load_pickle(cwd + "/models/" + model[0] + "_oversampled_classifier.pickle")
                        output_file = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + "_oversampled.txt", 'w')
                        # print("Predicting " + model[0] + "_oversampled...")
                    else:
                        classifier = load_pickle(cwd + "/models/" + model[0] + "_classifier.pickle")
                        output_file = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + ".txt", 'w')
                        # print("Predicting " + model[0] + "...")

                    predictions = []
                    for i in range(len(data)):
                        prediction = classifier.predict_proba([to_predict[i]])
                        predictions.append(prediction[0][1])

                    for i in range(len(data)):
                        sentence_id = data[i][0]
                        score = normalize(predictions, predictions[i])
                        output_file.write(str(sentence_id) + "\t" + float_to_str(score) + "\n")

                    output_file.close()

            if combine:
                scores = []
                m = 0

                for model in All_models:
                    if model[1]:
                        classifier = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + "_oversampled.txt",
                                          encoding='utf8')
                    else:
                        classifier = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + ".txt", encoding='utf8')

                    scores.append([])
                    for line in classifier:
                        parts = line.split("\t")
                        scores[m].append(float(parts[1]))

                    m = m + 1

                output_file = open(cwd + "/output/" + filename[:-4] + "_combined.txt", 'w')
                # print("Writing combined...")

                scores = np.transpose(scores)
                for i in range(len(scores)):
                    # score = min(scores[i])
                    # score = max(scores[i])
                    # score = np.median(np.transpose(scores)[i])
                    score = np.mean(scores[i])

                    output_file.write(str(i + 1) + "\t" + float_to_str(score) + "\n")

                output_file.close()

                for modelToRemove in All_models:
                    models = list(All_models)
                    models.remove(modelToRemove)
                    scores = []
                    m = 0

                    for model in models:
                        if model[1]:
                            classifier = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + "_oversampled.txt",
                                              encoding='utf8')
                        else:
                            classifier = open(cwd + "/output/" + filename[:-4] + "_" + model[0] + ".txt",
                                              encoding='utf8')

                        scores.append([])
                        for line in classifier:
                            parts = line.split("\t")
                            scores[m].append(float(parts[1]))

                        m = m + 1

                    if modelToRemove[1]:
                        output_file = open(
                            cwd + "/output/" + filename[:-4] + "-" + modelToRemove[0] + "_oversampled.txt", 'w')
                        # print("Writing -" + modelToRemove[0] + "_oversampled...")
                    else:
                        output_file = open(cwd + "/output/" + filename[:-4] + "-" + modelToRemove[0] + ".txt", 'w')
                        # print("Writing -" + modelToRemove[0] + "...")

                    scores = np.transpose(scores)
                    for i in range(len(scores)):
                        # score = min(scores[i])
                        # score = max(scores[i])
                        # score = np.median(np.transpose(scores)[i])
                        score = np.mean(scores[i])

                        output_file.write(str(i + 1) + "\t" + float_to_str(score) + "\n")

                    output_file.close()
