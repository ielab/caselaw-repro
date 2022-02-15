import os

HOME_DIR = os.environ["HOME"]
# BASE_DIR = '/Users/danlocke/go/src/github.com/dan-locke/phd/experiments/'
BASE_DIR = os.path.join(HOME_DIR, 'go/src/github.com/dan-locke/phd/experiments/')

# DATA_DIR = '/Users/danlocke/go/src/github.com/dan-locke/phd-data/'
DATA_DIR = os.path.join(HOME_DIR, 'go/src/github.com/dan-locke/phd-data/')


AUS_TOPIC_PATH = os.path.join(DATA_DIR, 'case-topics.json')
AUS_LEG_TOPIC_PATH = os.path.join(DATA_DIR, 'legislative-queries.json')

AUS_DATA_DIR = 'aus'
AUS_INDEX_NAME = 'flattened'
AUS_STEMMED_INDEX_NAME = 'flatstem'
AUS_QREL_PATH = os.path.join(DATA_DIR, AUS_DATA_DIR, 'filtered-qrels.txt')
AUS_REL_LEVEL = '1'
AUS_STOPWORD_PATH = os.path.join(DATA_DIR, AUS_DATA_DIR, 'qld-stopwords.txt')

SIGIR_DATA_DIR = 'sigir'
SIGIR_INDEX_NAME = 'sigir'
SIGIR_QREL_PATH = os.path.join(DATA_DIR, SIGIR_DATA_DIR, 'comb-sigir.txt')
SIGIR_REL_LEVEL = '1'

METRIC_NAMES = {
	'recip_rank': 'RR', 
    'err@20': 'ERR@20',
	'recall_20': 'R@20',
	'recall_100': 'R@100',
	'ndcg': 'NDCG',
	'rbp@0.80': 'RBP',
    'unjudged@20': 'Unjudged@20',
}

EXPANDED_METRIC_NAMES = {
	'recip_rank': 'RR', 
	'recall_20': 'R@20',
	'recall_100': 'R@100',
	'ndcg': 'NDCG',
	'rbp@0.80': 'RBP',
	'p': 'P',
	'R': 'R',
}

latex_args = {'float_format':"{:0.4f}".format, 'escape':False}