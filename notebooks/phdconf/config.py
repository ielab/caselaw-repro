import os

HOME_DIR = os.environ["HOME"]

BASE_DIR = os.path.join(os.getcwd(), "../run-files")
DATA_DIR = os.path.join(os.getcwd(), "../other")

AUS_TOPIC_PATH = os.getenv("AUS_TOPIC_PATH")
AUS_LEG_TOPIC_PATH = os.getenv("AUS_LEG_TOPIC_PATH")

AUS_DATA_DIR = 'aus'
AUS_REL_LEVEL = '1'
AUS_STOPWORD_PATH = os.path.join(DATA_DIR, AUS_DATA_DIR, 'qld-stopwords.txt')

SIGIR_DATA_DIR = 'sigir'
SIGIR_INDEX_NAME = 'sigir'
SIGIR_REL_LEVEL = '1'

AUS_QREL_PATH = os.getenv("AUS_QREL_PATH")
SIGIR_QREL_PATH = os.getenv("SIGIR_QREL_PATH")

AUS_FOLDS = os.path.join(os.getcwd(), '../eval-folds/ausnl-folds.txt')
SIGIR_FOLDS = os.path.join(os.getcwd(), '../eval-folds/sigir-folds.txt') 

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

RERANK_YLIMS = [0.67, 0.45, 0.57, 0.51, 0.38, 20]
ALL_YLIMS = [0.45, 0.75, 0.38]
YLIMS=[0.45, 0.38]

FULL_METRICS = {'recip_rank': 'RR', 'err@20': 'ERR@20', 'recall_20': 'R@20', 'recall_100': 'R@100', 'ndcg': 'NDCG', 'rbp@0.80': 'RBP', 'unjudged@20': 'Unjudged@20'}
RERANK_METRICS = {'recip_rank': 'RR', 'err@20': 'ERR@20', 'recall_20': 'R@20', 'ndcg': 'NDCG', 'rbp@0.80': 'RBP', 'unjudged@20': 'Unjudged@20'}
PAPER_FULL_METRICS = {'err@20': 'ERR@20', 'recall_100': 'R@100', 'rbp@0.80': 'RBP'}
PAPER_RERANK_METRICS = {'err@20': 'ERR@20', 'rbp@0.80': 'RBP'}

latex_args = {'float_format':"{:0.4f}".format, 'escape':False}