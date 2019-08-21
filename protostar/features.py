from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def term_frequency(text):

    text = text.lower()
    # text = processor.stem_words(text)
    words = word_tokenize(text)

    dictionary = {
        "dumb": 0.5,
        "muthafuckin": 0.4,
        "muthafucking": 0.4,
        "motherfucking": 0.4,
        "motherfucker": 0.4,
        "motherfuckers": 0.4,
        "moron": 0.5,
        "stupid": 0.5,
        "slaughter": 0.5,
        "massacre": 0.3,
        "genocide": 0.4,
        "criminal": 0.3,
        "thug": 0.4,
        "violent": 0.3,
        "faggot": 0.7,
        "sissy": 0.7,
        "fucking": 0.5,
        "disgusting": 0.6,
        "baboon": 0.4,
        "coward": 0.4,
        "foreigner": 0.3,
        "outsider": 0.3,
        "ugly": 0.5,
        "uncivilised": 0.5,
        "kaffir": 1.0,
        "kaffirs": 1.0,
        "kaffer": 1.0,
        "kaffers": 1.0,
        "kafer": 1.0,
        "kafers": 1.0,
        "kafir": 1.0,
        "kafirs": 1.0,
        "kuli": 0.8,
        "coolie": 0.7,
        "hoe": 0.7,
        "hoes": 0.6,
        "whore": 0.8,
        "whores": 0.7,
        "pig": 0.5,
        "fake": 0.4,
        "fong kong": 0.4,
        "sodomy": 0.4,
        "whitey": 0.6,
        "hottentot": 0.6,
        "chinaman": 0.6,
        "chink": 0.6,
        "chinkie": 0.6,
        "coon": 0.5,
        "half breed": 0.7,
        "hairyback": 0.7,
        "jungle bunny": 0.6,
        "kraut": 0.5,
        "goy": 0.5,
        "shiksa": 0.6,
        "yid": 0.6,
        "fag": 0.4,
        "dyke": 0.6,
        "hooligan": 0.6,
        "shithole": 0.6,
        "slave": 0.3,
        "bulldyke": 0.7,
        "muppet": 0.5,
        "lunatic": 0.5,
        "shoot": 0.4,
        "nansy pansy": 0.5,
        "terrorist": 0.4,
        "monkey": 0.3,
        "moffie": 0.7,
        "redneck": 0.7,
        "rednecks": 0.7,
        "kike": 0.7,
        "kikes": 0.6,
        "abomination": 0.5,
        "lawless": 0.4,
        "amakwerekwere": 0.6,
        "kwerekwere": 0.6,
        "ofay": 0.5,
        "mohammedan": 0.5,
        "lesbo": 0.5,
        "barbaric": 0.7,
        "barbarics": 0.7,
        "barbarians": 0.8,
        "barbarian": 0.8,
        "ape": 0.4,
        "carpet muncher": 0.7,
        "rag muncher": 0.7,
        "bigot": 0.5,
        "bigots": 0.5,
        "loot": 0.4,
        "thieves": 0.2,
        "steal": 0.4,
        "dirty": 0.5,
        "garden boy": 0.6,
        "trash": 0.6,
        "garbage": 0.6,
        "honky": 0.6,
        "nigger": 0.8,
        "negro": 0.7,
        "nigga": 0.6,
        "niggas": 0.8,
        "negros": 0.7,
        "niggers": 0.8,
        "bastard": 0.6,
        "bastards": 0.6,
        "gook": 0.6,
        "sand nigger": 0.7,
        "sand nigga": 0.7,
        "sand niggas": 0.7,
        "sand niggers": 0.7,
        "ghetto": 0.5,
        "ratchet": 0.5,
        "tranny": 0.7,
        "thot": 0.6,
        "cunt": 0.5,
        "sicko": 0.5,
        "sickos": 0.5,
        "coconut": 0.5,
        "amabujwas": 0.5,
        "mabujwa": 0.5,
        "clever blacks": 0.2,
        "oreo": 0.3,
        "coloniser": 0.3,
        "colonialist": 0.3,
        "colonizer": 0.3,
        "butthumper": 0.7,
        "istabane": 0.8,
        "stabane": 0.8,
        "sodomite": 0.7,
        "sodomis": 0.6,
        "trassie": 0.6,
        "sisBhuti": 0.7,
        "bhut'sisi": 0.7,
        "umanzi": 0.5,
        "trap": 0.3,
        "kill the boers": 0.8,
        "kill the boer": 0.8,
        "kill te boer": 0.6,
        "dubula ibhunu": 0.6,
        "shoot the boer": 0.7,
        "kill": 0.5,
    }
    columns = dictionary.keys()

    hate_count = 0

    dict_features = {}

    for dic_term in dictionary.keys():
        count = 0
        for word in words:
            if word == dic_term:
                count = count + 1
        hate_count = hate_count + count
        dict_features[dic_term] = count * dictionary[dic_term]

    dict_features["weight"] = hate_count / len(text)

    return dict_features


def get_tfidf_scores(text):

    with open("protostar/tfidf_vocab.pickle", "rb") as f:
        vocab = pickle.load(f)

        dic = {"tweet": text}
        df = pd.DataFrame(dic, index=[0])

        count_v = CountVectorizer(ngram_range=(1, 4), min_df=0.001, vocabulary=vocab)
        count_v.fit(df.tweet)
        count_matrix = count_v.transform(df.tweet)
        transformer = TfidfTransformer()
        tfidf_scores = transformer.fit_transform(count_matrix)

        tfidf_scores = pd.DataFrame(
            tfidf_scores.todense(), index=[0], columns=count_v.get_feature_names()
        )
        return tfidf_scores


def get_sentiment_score(text):

    analyser = SentimentIntensityAnalyzer()
    score = analyser.polarity_scores(text)
    return score
