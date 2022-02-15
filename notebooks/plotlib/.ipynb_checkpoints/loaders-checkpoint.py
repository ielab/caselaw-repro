import json
import os 

import numpy as np
from plotlib.trec_df import to_trec_df
from typing import Set 

def load_stopwords(path: str) -> Set:
    stop = set()
    with open(path) as f: 
        for line in f: 
            stop.add(line.strip())

    return stop 

def load_queries(path: str):  
    queries = {}
    with open(path) as f:
        data = json.load(f)
        for topic in data['topics']:
            leg_ref = None 
            if 'legislation_ref' in topic: 
                leg_ref = topic['legislation_ref']
            queries[topic['id']] = {'topic': topic['topic'], 'type': topic['type'], 'legislation_ref': leg_ref, 'focus': topic['focus']}
    return queries


def load_query_tf_file(path: str): 
    """ Loads query term frequency file """    
    queries = {}
    
    with open(path) as f: 
        for line in f:
            parts = line.strip().split()
            cnts = {}
            for i in range(1, len(parts), 2):
                cnts[parts[i]] = cnts.get(parts[i], 0) + int(parts[i+1])
                
            queries[parts[0]] = cnts
        
    return queries

def load_query_types(queries):
    b, s = [], []
    for k, v in queries.items():
        if v['type'] == 'specific': 
            s.append(k)
        else:
            b.append(k)
    return b, s 

def load_query_focus_types(queries):
    l, f, g = [], [], []
    for k, v in queries.items():
        t = v['focus']
        if t == 'law': 
            l.append(k)
        elif t == 'fact':
            f.append(k)
        else: 
            g.append(k)
    return l, f, g 


def load_1d_dfs(index_names, qrel_paths, results_path, run_format, rel_levels, start, end, increment, per_query=False, filtered=None):
    dfs = []
    iterator = np.arange(start, end+increment, increment)
    # deal with shitty float overflow on np.arange 
    if iterator[-1] > end: 
        iterator = iterator[:-1]
    for i, ind in enumerate(index_names):
        temp = []
        for l in iterator:
            temp.append(to_trec_df(qrel_paths[i], os.path.join(results_path, run_format.format(ind, l)), rel_levels[i], per_query, filtered))
        dfs.append(temp)
    
    return dfs 


def load_dfs(qrel_path, rel_level, results_path, names, per_query=False, filtered=None):
    dfs = []
    for n in names:
        dfs.append(to_trec_df(qrel_path, os.path.join(results_path, n), rel_level, per_query, filtered))
    
    return dfs 


def load_doclen_lookup(path: str): 
	out = {}
	with open(path) as f:
		for line in f: 
			parts = line.split()
			if len(parts) == 2:
				out[parts[0]] = int(parts[1])
				
	return out 