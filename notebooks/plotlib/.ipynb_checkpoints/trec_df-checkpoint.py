import os 
import subprocess
import pandas as pd

GDEVAL_PATH = os.environ["HOME"]

def process(args): 
    res = subprocess.check_output(args)
    results = {}

    for line in res.decode('utf-8').split('\n'):
        parts = line.split()
        if len(parts) == 3:
            if parts[1] != "all":
                qry = int(parts[1])
                if qry not in results: 
                    results[qry] = {}
                if parts[0] == 'relstring_20':
                    results[qry]['unjudged@20'] = float(parts[2].count('-'))
                else:
                    results[qry][parts[0]] = float(parts[2])
                
    return results  

def process_rbp(args): 
    res = subprocess.check_output(args)
    results = {}
    
    for line in res.decode('utf-8').split('\n'):
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 9 or parts[3] == 'all':
            continue
            
        q_id = int(parts[3])    
        if q_id not in results:
            results[q_id] = {} 
        
        results[q_id]['rbp@'+parts[1]] = float(parts[7])
        results[q_id]['rbp-res@'+parts[1]] = float(parts[8][1:])

    return results

def process_gdeval(args):
    res = subprocess.check_output(args)
    results = {}
    lines = res.decode('utf-8').split('\n')
    depth = lines[0].split(',')[3]
    for line in lines[1:]:
        parts = line.split(',')
        if parts[1] == 'amean':
            break
        q_id = int(parts[1])    
        if q_id not in results:
            results[q_id] = {} 
        
        results[q_id][depth] = float(parts[3])

    return results


def to_trec_df(qrel_path: str, res_path: str, rel_level='1', per_query=False, filtered=None): 
    results = process(['trec_eval', '-q', '-m', 'recall.20,100', '-m', 'ndcg', '-m', 'recip_rank', '-m', 'set_P', '-m', 'set_recall', '-m', 'relstring.20', '-m', 'set_F..5', '-l', rel_level, qrel_path, res_path])
    df = pd.DataFrame.from_dict(results, orient='index')

    rbp_res = process_rbp(['rbp_eval', '-q', '-d', '100', '-p', '0.1,0.5,0.8', '-b', rel_level, qrel_path, res_path])
    
    err_res = process_gdeval(['perl', os.path.join(GDEVAL_PATH, 'gdeval.pl'), qrel_path, res_path])
    
    rbp_df = pd.DataFrame.from_dict(rbp_res, orient='index')
    err_df = pd.DataFrame.from_dict(err_res, orient='index')
    
    df = df.join(rbp_df)
    df = df.join(err_df)
    
    if filtered: 
        df = df[df.index.isin(filtered)]
        
    if per_query:
        return df
    
    return df.mean(axis=0)
    