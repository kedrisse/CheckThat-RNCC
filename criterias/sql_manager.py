import sqlite3


def min_max_score_to_dict(min_max_scores):
    return {
        'factuality': {'min': min_max_scores[0], 'max': min_max_scores[1]},
        'readability': {'min': min_max_scores[2], 'max': min_max_scores[3]},
        'emotion': {'min': min_max_scores[4], 'max': min_max_scores[5]},
        'opinion': {'min': min_max_scores[6], 'max': min_max_scores[7]},
        'controversy': {'min': min_max_scores[14], 'max': min_max_scores[15]},
        'trust': {'min': min_max_scores[8], 'max': min_max_scores[9]},
        'technicality': {'min': min_max_scores[10], 'max': min_max_scores[11]},
        'topicality': {'min': min_max_scores[12], 'max': min_max_scores[13]},
    }


class SQLManager:

    def __init__(self):
        # Connecting to the database file
        filename = 'criterias/db.sqlite3'
        conn = sqlite3.connect(filename)
        c = conn.cursor()
        self.db_connector = conn
        self.db_cursor = c

        if not self.table_exists():
            self.create_score_table()

    def create_score_table(self):
        self.db_cursor.execute(
            'CREATE TABLE scores(id INT PRIMARY KEY, article_url TEXT UNIQUE, factuality REAL, '
            'readability REAL, emotion REAL, opinion REAL, controversy REAL, trust REAL, technicality REAL, '
            'topicality REAL, factuality_desc TEXT, readability_desc TEXT, emotion_desc TEXT, '
            'opinion_desc TEXT, controversy_desc TEXT, trust_desc TEXT, technicality_desc TEXT, '
            'topicality_desc TEXT)')

    def insert_new_scores(self, article_url, scores):
        self.db_cursor.execute(
            'INSERT INTO scores '
            'VALUES ((SELECT count(id)+1 FROM scores), \'{article_url}\', {factuality}, {readability}, {emotion}, {opinion}, {controversy}, {trust}, '
            '{technicality}, {topicality}, {factuality_desc}, {readability_desc}, {emotion_desc}, {opinion_desc}, {controversy_desc}, {trust_desc}, {technicality_desc}, {topicality_desc})'
                .format(article_url=article_url,
                        factuality=scores['factuality']['score'] if scores['factuality']['score'] is not None else 'null',
                        readability=scores['readability']['score'] if scores['readability']['score'] is not None else 'null',
                        emotion=scores['emotion']['score'] if scores['emotion']['score'] is not None else 'null',
                        opinion=scores['opinion']['score'] if scores['opinion']['score'] is not None else 'null',
                        controversy=scores['controversy']['score'] if scores['controversy']['score'] is not None else 'null',
                        trust=scores['trust']['score'] if scores['trust']['score'] is not None else 'null',
                        technicality=scores['technicality']['score'] if scores['technicality']['score'] is not None else 'null',
                        topicality=scores['topicality']['score'] if scores['topicality']['score'] is not None else 'null',
                        factuality_desc='\''+scores['factuality']['desc']+'\'' if scores['factuality']['desc'] is not None else 'null',
                        readability_desc='\''+scores['readability']['desc']+'\'' if scores['readability']['desc'] is not None else 'null',
                        emotion_desc='\''+scores['emotion']['desc']+'\'' if scores['emotion']['desc'] is not None else 'null',
                        opinion_desc='\''+scores['opinion']['desc']+'\'' if scores['opinion']['desc'] is not None else 'null',
                        controversy_desc='\''+scores['controversy']['desc']+'\'' if scores['controversy']['desc'] is not None else 'null',
                        trust_desc='\''+scores['trust']['desc']+'\'' if scores['trust']['desc'] is not None else 'null',
                        technicality_desc='\''+scores['technicality']['desc']+'\'' if scores['technicality']['desc'] is not None else 'null',
                        topicality_desc='\''+scores['topicality']['desc']+'\'' if scores['topicality']['desc'] is not None else 'null'))
        self.db_connector.commit()

    def table_exists(self):
        self.db_cursor.execute('SELECT name FROM sqlite_master WHERE type=\'table\' AND name=\'scores\'')
        return self.db_cursor.fetchone() is not None

    def article_exists(self, url):
        self.db_cursor.execute('SELECT count(id) FROM scores WHERE article_url=\'{}\''.format(url))
        return self.db_cursor.fetchone()[0] == 1

    def get_scores(self, article_url):
        self.db_cursor.execute('SELECT * FROM scores WHERE article_url=\'{}\''.format(article_url))
        return self.db_cursor.fetchone()

    def get_min_max_scores(self):
        self.db_cursor.execute('SELECT MIN(factuality), MAX(factuality), MIN(readability), MAX(readability), '
                               'MIN(emotion), MAX(emotion), MIN(opinion), MAX(opinion), MIN(trust), MAX(trust), '
                               'MIN(technicality), MAX(technicality), MIN(topicality), MAX(topicality), '
                               'MIN(controversy), MAX(controversy) FROM scores')
        return min_max_score_to_dict(self.db_cursor.fetchone())

    def save(self):
        self.db_connector.commit()
        self.db_connector.close()
