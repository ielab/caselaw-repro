#!/usr/bin/env python3.7

from functools import partial
import http.server
import json
import os
import pickle
import subprocess
import socketserver
import sys
import argparse
from typing import List

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer
from index.knn import AnnoyKnnIndex, KnnIndex

PORT = 9201

DEFAULT_K = 10

INDEX_NAME = 'vec-ind'

PARSE_PATH = "/home/danlocke/go/src/parse/./parse"

def to_id(t: str) -> str:
        if '-' in t:
                parts = t.split('-', -1)
        else:
            parts = []
            start = 0
            curr = True
            t_len = len(t)
            for i in range(t_len):
                    past = curr
                    if t[i].isdigit():
                        curr = True
                    else:
                        curr = False

                    if past != curr:
                        parts.append(t[start:i])
                        start = i

                    if i+1 == t_len:
                        parts.append(t[start:])

        for i in range(len(parts)):
                if parts[i].isdigit():
                        parts[i] = str(int(parts[i]))

        return ''.join(parts)

class EmbeddingMethod:
        SUM = 0
        MEAN = 1

class EmbeddingLookup:

        def __init__(self, method:EmbeddingMethod = EmbeddingMethod.SUM):
                self.lookup = {}
                self._method = method
                self._dim = -1

        def from_file(self, path: str): 
                with open(path) as f:
                        first_line = f.readline().split()
                        self._dim = int(first_line[1])
                        for line in f:
                                parts = line.split()
                                vec = [] 
                                for v in parts[1:]:
                                        vec.append(float(v))

                                if len(vec) != self._dim:
                                        print("Inconsistent lookup dimensions: {0} with dim {1} and standard dim {2}".format(k, v, dim))
                                self.lookup[parts[0]] = np.array(vec)

        def to_vector_nd(self, tokens: List[List[str]]) -> np.array:
                vec = np.zeros((len(tokens), self._dim,), dtype=np.float32)

                for i, para in enumerate(tokens):
                        for j, t in enumerate(para):
                                if t in self.lookup: 
                                        vec[i] += self.lookup[t]

                        if self._method == EmbeddingMethod.MEAN:
                                vec[i] /= len(para)
                
                return vec

        def to_vector(self, tokens: List[str]) -> np.array:
                vec = np.zeros((self._dim,), dtype=np.float32)

                for t in tokens:
                        if t in self.lookup:
                                vec += self.lookup[t]

                if self._method == EmbeddingMethod.MEAN:
                        vec /= len(tokens)

                return vec

exclude = {
    "<i>", 
    "</i>",
    "<br>",
    "<b>",
    "</b>",
}

def create_index(path: str, lookup: EmbeddingLookup, sen, clean, bert, act: bool, field: str = 'text') -> KnnIndex:
        dim = 768 if bert else 100
        index = AnnoyKnnIndex(dim)
        embedder = None 
        if bert:
            embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
#        all_paras = []
        for root, d, files in os.walk(path):
                for f in files:
                        if f.endswith('.json'):
                            if act:
                                with open(os.path.join(root, f)) as _f:
                                    print(f)
                                    data = json.load(_f)
                                    if 'body' not in data:
                                        continue
                                    for item in data['body']:
                                        if item[field] == "": 
                                            continue
                                        res = subprocess.run(['/home/danlocke/go/src/parse-query/./parse-query', item[field]], stdout=subprocess.PIPE).stdout
                                        parsed = json.loads(res)
                                        if len(parsed) == 0:
                                            continue
                                        index.add(lookup.to_vector_nd(parsed['tokens']), '{0}-{1}'.format(f[:-5], item['pos']))
                            else:
                                if clean:
                                    res = subprocess.run([PARSE_PATH, os.path.join(root, f)], stdout = subprocess.PIPE).stdout
                                    data = json.loads(res)
                                    doc_id = to_id(f[:-5])
                                    print(doc_id)
                                    paras = []
                                    for p in data["paras"]:
                                            if len(p) > 0:
                                                    p = [x.lower() for x in p if x not in exclude] 
                                                    paras.append(p)
    #                                                all_paras.append(p)
                                    if len(paras) > 0:
                                        if bert:
                                            paras = [' '.join(x) for x in paras]
                                            enc = embedder.encode(paras)
                                            index.add(enc, doc_id)
                                        else:
                                            index.add(lookup.to_vector_nd(paras), doc_id)
                                else:
                                    with open(os.path.join(root, f)) as _f:
                                        data = json.load(_f)
                                        doc_id = to_id(f[:-5])
                                        print(doc_id)
                                        paras = [x['text'] for x in data['body'] if x['type'] == 'paragraph' or x['type'] == 'quote']
                                        sens = []
                                        if args.sen: 
                                            for x in paras:
                                                sens += nltk.sent_tokenize(x)
                                            paras = sens
                                        if paras is None or len(paras) == 0:
                                            continue
                                        enc = embedder.encode(paras)
                                        index.add(enc, doc_id)


        index.create_index()
        return index #, all_paras

class SearchMethod:
        TOP = 0
        # TODO 
        COUNT = 1 

class SearchIndex:

        def __init__(self, index: KnnIndex, lookup: EmbeddingLookup, search_method: SearchMethod = SearchMethod.TOP, act:bool = False, bert:bool=False):
                self._index = index
                self._bert = bert
                self._encoder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens') if self._bert else None
                self._lookup = lookup
                self._search_method=search_method
                self._act = act

        def search(self, qry: List[str], k=100, data=None):
                if self._bert:
                    qry_vec = self._encoder.encode(' '.join(qry))[0]
                else:
                    qry_vec = lookup.to_vector(qry)
                
                cnt = 1
                num_docs = 0
                seen_ids = set()
                scores = []
                if self._search_method == SearchMethod.TOP:
                        if self._act:
                                res_ids, res_rows, res_scores, self._index.search(qry_vec, k*cnt, data)
                                for i in range(len(res_ids)):
                                        scores.append((res_ids[i], res_rows[i], 1.0-float(res_scores[i])))
                                return scores

                        while num_docs < k:
                                res_ids, res_rows, res_scores = self._index.search(qry_vec, k*cnt, data)
                                for i in range(len(res_ids)):
                                        if res_ids[i] not in seen_ids:
                                                seen_ids.add(res_ids[i])
                                                num_docs+=1
                                                scores.append((res_ids[i], int(res_rows[i]), 1.0 - float(res_scores[i])))
                                cnt+=1 
                                if cnt > 1:
                                        break

                return scores

        def get_docs_scores(self, qry: List[str], docs: List[str]):
                scores = []
                qry_vec = lookup.to_vector(qry)
                for doc in docs:
                        scores.append(self._get_doc_score(qry_vec, doc))
                return scores

        def _get_doc_score(self, qry_vec: np.array, doc: str):
                doc_rows = self._index.get_doc_rows(doc)
                return self._index.get_max_sim_rows(qry_vec, doc_rows) 

        def get_doc_score(self, qry: List[str], doc: str):
                qry_vec = lookup.to_vector(qry)
                doc_rows = self._index.get_doc_rows(doc)
                return self._index.get_max_sim_rows(qry_vec, doc_rows) 

class IndexRequestHandler(http.server.BaseHTTPRequestHandler):

        def __init__(self, index: SearchIndex, *args, **kwargs):
                self._index = index
#                self._data = data
                super().__init__(*args, **kwargs)

        def do_POST(self):
                content_length = int(self.headers['Content-Length'])
                body = self.rfile.read(content_length)
                print(body)
                self.send_response(200)
                
                data = json.loads(body)
                q_type = data.get("q_type", None)
                search_res = None
                if q_type is not None: 
                        if q_type == "score":
                                search_res = self._index.get_docs_scores(data["toks"], data["docs"])
                        elif q_type == "search":
                                search_res = self._index.search(data["toks"], k=data.get("k", DEFAULT_K))#, data=self._data)

                res = json.dumps({"results": search_res})
                self.end_headers()
                self.wfile.write(res.encode('utf-8'))


if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--serve', dest='serve', action='store_true')
        parser.add_argument('--act', dest='act', action='store_true')
        parser.add_argument('--title', dest='title', action='store_true')
        parser.add_argument('--clean', dest='clean', action='store_true')
        parser.add_argument('--sen', dest='sen', action='store_true')
        parser.add_argument('--bert', dest='bert', action='store_true')
        parser.add_argument('-i', dest='index_name', default=INDEX_NAME)
        parser.add_argument('-f', dest='files', default='')
        parser.add_argument('-e', dest='emb_path', default='')
        parser.set_defaults(feature=True)

        args = parser.parse_args()
        lookup = None

        if not args.bert:
                if args.emb_path == '':
                        print('Please specify either BERT embedding flag --bert or path to embeddings -f')
                        sys.exit(1)

                lookup = EmbeddingLookup(EmbeddingMethod.MEAN)
                lookup.from_file(args.emb_path)
        
        index = None

        if not args.serve:
                if args.files == '':
                        print("Provide path to files to index")
                        sys.exit(1)
                
                field = 'text'
                if args.title:
                        field = 'title'

                index = create_index(args.files, lookup, args.sen, args.clean, args.bert, args.act, field)

                index.save(args.index_name)
                sys.exit(0)

  
        if args.index_name == '':
                print('Must provide index name to serve')

        if args.bert:
                index = AnnoyKnnIndex(768)
        else:
                index = AnnoyKnnIndex()
        index.load(args.index_name)
        print('Index contains {0} docs.'.format(len(set([x[2] for x in index.id_lookup]))))
        search_index = SearchIndex(index, lookup, act=args.act, bert=args.bert)
        
#        data = None
#        with open('dat.pkl', 'rb') as f:
#            data = pickle.load(f)

        handler = partial(IndexRequestHandler, search_index) #, data)

        with socketserver.TCPServer(("", PORT), handler) as httpd:
                print("serving at port", PORT)
                httpd.serve_forever()    
