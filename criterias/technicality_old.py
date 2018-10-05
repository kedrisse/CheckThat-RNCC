#!/usr/bin/env python
# -*- coding: utf-8 -*-

import nltk
import re


# NP = groupes de mots techniques
def traitementArticle(texte):
    # rajoute des espaces avant et après chaque signe de ponctuation
    # puis renvoie la liste des couples des mots associés à leur tag

    ponctuation = [",", '"', ".", ";", ":", "(", ")", "?", "!", "-"]
    for elt in ponctuation:
        texte = texte.replace(elt, " " + elt + " ")
    texte = texte.replace("  ", " ")
    texte = texte.replace("  ", " ")

    words = nltk.word_tokenize(texte)
    listeCoupleMotTag = nltk.pos_tag(words)
    return listeCoupleMotTag


def getPattern():
    # renvoie le pattern utilisé pour trouver les termes techniques

    regexpr = "(((JJ |NN )+|((JJ |NN )+(IN )?)(JJ |NN )+)NN)"
    pattern = re.compile(regexpr)
    return pattern


def traitementRegExpr(listeCoupleMotTag):
    # renvoie une string composée des tags des mots
    # ainsi qu'une liste des termes valide (composée que des tags associés aux mots)
    # correspondant au pattern utilisé

    pattern = getPattern()
    stringTags = ""

    for tag in listeCoupleMotTag:
        if (tag[1] not in ["NN", "JJ", "IN", "NNS"]):
            stringTags += "XX" + " "
        else:
            stringTags += tag[1] + " "
    stringTags = stringTags[:len(stringTags) - 1]
    stringTags = stringTags.replace("NNS", "NN")

    regExprTrouvees = re.findall(pattern, stringTags)

    regExprValides = []

    for regExprTrouvee in regExprTrouvees:
        regExprValides.append(regExprTrouvee[0])

    return regExprValides, stringTags


def traitementCritere(listeCoupleMotTag, regExprValides, stringTags):
    # pour chaque regExpr valide, la fonction récupère les mots dans la string
    # correspondant aux tags
    # crée ensuite le dictionnaire qui a en clés les termes trouvés
    # et en valeur le nombre de fois qu'ils sont apparus dans le texte

    cpt = 0
    dico = {}

    while (stringTags != "" and regExprValides != []):
        regExprCourante = regExprValides[0]
        # indice de string du debut de la regExprValide
        indice = stringTags.find(regExprCourante)
        # liste de doublet pour recuperer les mots
        # correspondant aux listeCoupleMotTag
        listeCoupleTrouvee = listeCoupleMotTag[
                             int((indice + cpt) / 3):int((indice + cpt) / 3 + len(regExprCourante.split(" ")))]
        # liste apres suppression jusqu'a la fin de la regExprValide
        stringTags = stringTags[indice + len(regExprCourante) + 1:]

        NP = ""
        for couple in listeCoupleTrouvee:
            NP += couple[0] + " "
        NP = NP[:len(NP) - 1]

        dico = traitementDico(dico, regExprCourante, NP)

        cpt += (indice + len(regExprCourante) + 1)
        regExprValides = regExprValides[1:]

    return dico


def traitementDico(dico, regExprCourante, NP):
    # rajoute la regExprCourante au dico
    # et renvoie ce dernier

    pattern = getPattern()
    NPsplit = NP.split(" ")
    regExprCourante2 = regExprCourante[:]
    while (len(NPsplit) > 1):
        NPreduit = " ".join(NPsplit)
        if (re.match(pattern, regExprCourante) != None):
            if (NPreduit in dico):
                dico[NPreduit] += 1
            else:
                dico[NPreduit] = 1
        NPsplit = NPsplit[1:]
        regExprCourante2 = regExprCourante2[3:]
    return dico


def score(texte):
    # calcule le score (en %) de technicalité : nbMotsTechniques/nbMotsTotal

    listeCoupleMotTag = traitementArticle(texte)
    regExprValides, stringTags = traitementRegExpr(listeCoupleMotTag)
    dico = traitementCritere(listeCoupleMotTag, regExprValides, stringTags)
    l = []
    for key in dico:
        if (dico[key] == 1):
            l.append(key)

    for elt in l:
        del dico[elt]

    score = 0
    for key in dico:
        score += dico[key]

    return score * 100. / len(texte.split(" "))


# C'est la fonction score qu'il faut appeler, elle permet de tout faire
# il faut qu'elle prenne en paramètre le texte qu'on récupère de l'article
# fic = open("article_test2", "r")
# contenu = fic.read()
#
# print("score", score(contenu))

class Technicality:

    def __init__(self, text):
        self.text = text

    def get_score(text):
        return score(text)

# ---