from typing import Dict, List, Union

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib.artist import setp

from IPython.display import set_matplotlib_formats

import numpy as np
import pandas as pd

from scipy import stats 

import seaborn as sns

from plotlib.trec_df import to_trec_df
from plotlib.loaders import load_dfs

import copy

from itertools import chain

set_matplotlib_formats('pdf')

#Set general plot properties
sns.set()
sns.set_context("paper")
sns.set_context({"figure.figsize": (16, 10)})
sns.set_color_codes("pastel")

plt.style.use('seaborn-white')

def plot_tune_1d(index_names, metric_names, df, start, end, increment): 
    fig, axs = plt.subplots(len(metric_names), len(index_names))
    fig.set_size_inches(16, 10)
    for i in range(len(index_names)):
        cnt = 0 
        for j, m in enumerate(df[i][0].index):
            if m in metric_names: 
                y = [y[m] for y in df[i]]
                axs[cnt, i].plot(np.arange(start, end, increment), y)

                if i == 0:
                    axs[cnt, i].set_ylabel(metric_names[m],fontsize=18)

                axs[cnt, i].tick_params(labelsize=12)
                axs[cnt, i].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
                cnt += 1
                
        axs[cnt-1, i].set_xlabel(index_names[i], fontsize=20)
        
    fig.tight_layout()
    return fig 

def plot_tune_1d_comp(names, metric_names, dfs, start, end, increment, legend_x: float=0.96, legend_y: float=0.46, styles=[], ylims=[]): 

    r = int(len(metric_names)/2)
    c = r
    if c == r: 
        r-=1
    if len(metric_names)%2 != 0:
        c += 1 
    fig, axs = plt.subplots(r, c)
    fig.set_size_inches(16, 6)
    x = np.arange(start, end+increment, increment)
    cnt = 0 
    row = 0
    for m in metric_names:
            for i, df in enumerate(dfs):
                s = None 
                if i < len(styles): 
                    s = styles[i]
                
                axs[row, cnt].plot(x, [y[m] for y in df], linestyle=s)
                if m.startswith('rbp@'):
                    es = 'rbp-res@'+m[4:]
                    axs[row, cnt].fill_between(x, [y[m] for y in df], [y[es]+y[m] for y in df], alpha=0.3)

            axs[row, cnt].set_ylabel(metric_names[m],fontsize=18)

            axs[row, cnt].tick_params(labelsize=12)
            axs[row, cnt].yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
            cnt += 1 
            if cnt >= c: 
                cnt = 0 
                row += 1 
                
    for i in range(len(ylims)):
        plt.gcf().get_axes()[i].set_ylim(ymax=ylims[i])
    
    if len(metric_names) % 2 != 0: 
        fig.delaxes(axs[row, -1])

    fig.legend(names, bbox_to_anchor=[legend_x, legend_y], frameon=True, ncol=2, prop={"size": 15}).get_frame().set_edgecolor('black')
        
    fig.tight_layout()
    return fig

# def plot_diff_query_ind(a, b, ind, metric_names, queries, decreased_df=None):
#     df = a[ind] - b[ind]
#     df['type'] = pd.Series({k: queries[k]['type'] for k in df.index})

#     for k, v in metric_names.items():
#         sort = df.sort_values([k], ascending=True)
#         mask = sort['type'] == 'broad'
#         colors = np.array(['b']*len(sort))
#         colors[mask.values] = 'r'
#         fig = plt.figure() 
#         ax = fig.add_subplot(111)
#         fig.set_size_inches(16, 10)
#         sns.barplot(x=df.index, y=k, data=df, order=sort.index, ax=ax, palette=colors, alpha=0.7)
#         ax.set_ylabel(v, fontsize=15)
#         labels = [x if sort[k].loc[x] != 0.0 else '' for x in sort.index] 
#         lab_colors = ['k' for x in sort.index]
#         if decreased_df is not None:
#             decreased = decreased_df[decreased_df[k] < 0]
#             ticks = ax.xaxis.get_ticklabels()
#             for x in decreased.index:
#                 ind = sort.index.get_loc(x)
#                 labels[ind] = x 
#                 ticks[ind].set_color('r')  
#         ax.set_xticklabels(labels, rotation=90, fontsize=11)
        
#         fig.tight_layout()       
        
        
def plot_diff_query_ind(a, b, ind, metric_names, queries, focus=False, decreased_df=None):
    df = a[ind] - b[ind]
    if focus:
        df['type'] = pd.Series({k: queries[k]['focus'] for k in df.index})
    else:
        df['type'] = pd.Series({k: queries[k]['type'] for k in df.index})

    for k, v in metric_names.items():
        sort = df.sort_values([k], ascending=True)
        if focus: 
            types = set([x['focus'] for x in queries.values()])
            mask = sort['type'] == 'law'
            mask2 = sort['type'] == 'generic'
            colors = np.array(['b']*len(sort))
            colors[mask.values] = 'r'
            colors[mask2.values] = 'g'
        else:
            mask = sort['type'] == 'broad'
            colors = np.array(['b']*len(sort))
            colors[mask.values] = 'r'
        fig = plt.figure() 
        ax = fig.add_subplot(111)
        fig.set_size_inches(16, 10)
        sns.barplot(x=df.index, y=k, data=df, order=sort.index, ax=ax, palette=colors, alpha=0.7)
        ax.set_ylabel(v, fontsize=15)
        labels = [x if sort[k].loc[x] != 0.0 else '' for x in sort.index] 
        lab_colors = ['k' for x in sort.index]
        if decreased_df is not None:
            decreased = decreased_df[decreased_df[k] < 0]
            ticks = ax.xaxis.get_ticklabels()
            for x in decreased.index:
                ind = sort.index.get_loc(x)
                labels[ind] = x 
                ticks[ind].set_color('r')  
        ax.set_xticklabels(labels, rotation=90, fontsize=11)
        
        fig.tight_layout()       

def plot_diff(a, b, metric_names, queries=None):
    df = a - b
    if queries is not None: 
        df['type'] = pd.Series({k: queries[k]['type'] for k in df.index})

    r = int(len(metric_names)/2)
    c = r + 1
    fig, axs = plt.subplots(r, c)
    fig.set_size_inches(16, 10)
    cnt = 0 
    row = 0

    for k, v in metric_names.items():
        sort = df.sort_values([k], ascending=True)
        mask = sort['type'] == 'broad'
        colors = np.array(['b']*len(sort))
        colors[mask.values] = 'r'

        g = sns.barplot(x=df.index, y=k, data=df, order=sort.index, ax=axs[row, cnt], palette=colors, alpha=0.7)
#         g.set_xticklabels([x if sort[k].loc[x] != 0.0 else '' for x in sort.index ])
        g.set_xticklabels([])
        axs[row, cnt].set_ylabel(v, fontsize=15)
#         pos = axs[row, cnt].get_position()
#         axs[row, cnt].text(pos.x0+0.1, pos.y1-0.15, 'Broad+ {0}'.format(inc_b), fontsize=12)
#         axs[row, cnt].text(0.15, 0.8, 'Broad- {0}'.format(dec_b), fontsize=12)
        
#         axs[row, cnt].text(0.15, 0.7, 'Specific+ {0}'.format(inc_s), fontsize=12)
#         axs[row, cnt].text(0.15, 0.6, 'Specific- {0}'.format(dec_s), fontsize=12)
#         plt.yticks(size=15)
#         plt.xticks(rotation=90, size=12)
        fig.tight_layout()
        
        cnt += 1 
        if cnt >= c: 
            cnt = 0 
            row += 1 
       
    if len(metric_names) % 2 != 0: 
        fig.delaxes(axs[row, -1])


def select_1d_max(display_names, metric_names, dfs, start, increment, name, metrics=None):
    measure_max = {}
    for i in range(len(display_names)):
        for j in range(len(dfs[i])):
            for m in dfs[i][j].index:
                if m not in metrics: 
                    continue 
                val = dfs[i][j][m]
                if (display_names[i], metrics[m]) not in measure_max: 
                    measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}
                else: 
                    if measure_max[(display_names[i], metrics[m])]['-'] < val:
                        measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}

    max_df = pd.DataFrame.from_dict(measure_max).stack().unstack(level=0)
    return max_df.reindex(list(metrics.values()))

def select_1d_max_stat_sig(display_names, dfs, start, increment, name, base_qry, base_df, base_val, path, qrel_path, rel_level, metrics=None):
    measure_max = {}
    for i in range(len(display_names)):
        for j in range(len(dfs[i])):
            for m in dfs[i][j].index:
                if m not in metrics: 
                    continue 
                val = dfs[i][j][m]
                if (display_names[i], metrics[m]) not in measure_max: 
                    measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}
                else: 
                    if measure_max[(display_names[i], metrics[m])]['-'] < val:
                        measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}
    
    back_metric = {v: k for k, v in metrics.items()}
    for k, v in measure_max.items():
        if k[1] == 'Unjudged@20':
            continue
        _l = float(v[name])
        if _l == 0.00:
            v['-'] = '{0:.4f}'.format(v['-'])
        else:
            comp = load_dfs(qrel_path, rel_level, '', [path.format(k[0], float(v[name]))], per_query=True)[0]
            p = stats.ttest_rel(base_qry[back_metric[k[1]]], comp[back_metric[k[1]]]).pvalue
            if p < 0.01:
                v['-'] = '{0:.4f}'.format(v['-'])+'$^{**}$'
            elif p < 0.05:
                v['-'] = '{0:.4f}'.format(v['-'])+'$^{*}$'
            else:
                v['-'] = '{0:.4f}'.format(v['-'])
                
    
    for x in base_df.items():
        if x[0] not in metrics: 
            continue
        measure_max[('base', metrics[x[0]])] = {'-': '{0:.4f}'.format(x[1]), name: '{0:.2f}'.format(base_val)}
        
    max_df = pd.DataFrame.from_dict(measure_max).stack().unstack(level=0)
    return max_df.reindex(list(metrics.values()))

def select_1d_max_with_interp(display_names, dfs, start, increment, name, interp, base_qry, base_df, base_val, path, qrel_path, rel_level, metrics=None):
    measure_max = {}
    for i in range(len(display_names)):
        for j in range(len(dfs[i])):
            for m in dfs[i][j].index:
                if m not in metrics: 
                    continue 
                val = dfs[i][j][m]
                if (display_names[i], metrics[m]) not in measure_max: 
                    measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}
                else: 
                    if measure_max[(display_names[i], metrics[m])]['-'] < val:
                        measure_max[(display_names[i], metrics[m])] = {'-': val, name: '{0:.2f}'.format(j*increment+start)}

    back_metric = {v: k for k, v in metrics.items()}
    for k, v in measure_max.items():
        if k[1] == 'Unjudged@20':
            continue
        _l = float(v[name])
        if _l == 0.00:
            v['-'] = '{0:.4f}'.format(v['-'])
        else:
            interp.interpolate(path.format(k[0]), _l, 'tmp.run')
            comp = load_dfs(qrel_path, rel_level, '', ['tmp.run'], per_query=True)[0]
            p = stats.ttest_rel(base_qry[back_metric[k[1]]], comp[back_metric[k[1]]]).pvalue
            if p < 0.01:
                v['-'] = '{0:.4f}'.format(v['-'])+'$^{**}$'
            elif p < 0.05:
                v['-'] = '{0:.4f}'.format(v['-'])+'$^{*}$'
            else:
                v['-'] = '{0:.4f}'.format(v['-'])
                
        
    for x in base_df.items():
        if x[0] not in metrics: 
            continue
        measure_max[('base', metrics[x[0]])] = {'-': '{0:.4f}'.format(x[1]), name: '{0:.2f}'.format(base_val)}
        
    max_df = pd.DataFrame.from_dict(measure_max).stack().unstack(level=0)
    return max_df.reindex(list(metrics.values()))


def compute_stat_sig(files, qrel_path: str, comparators: str, metrics=None, filtered=None, rel_level: str='1'):
    
    df_ev = dict([(run_name, to_trec_df(qrel_path, res_path, rel_level, filtered=filtered)) for run_name, res_path, filtered in files])

    d = pd.DataFrame()
    # overall avg table, index is run_name
    # df_ev is mean (dict)
    df_results = pd.DataFrame(df_ev).T
    t = df_results.copy()
    # run name: tf_data_frame 
    # index is run_name
    
    df_eval = dict([(run_name, to_trec_df(qrel_path, res_path, rel_level, True)) for run_name, res_path, filtered in files])
    df = t.T
    df2 = t.T
    strs = []
    # %%
    s = {}
    symbols = ['*', '\dagger', '\textdollar', '\textyen', '\textcent', '\texteuro', '\text', '\cdot', '\circ', '\centerdot', '\diamond']
    # for each row
    for c, comp in enumerate(comparators):
        cnt = 0
        for x in df:
            if cnt < c: 
                cnt+=1
                continue
            # for each column
            for i in df[x].index:
                p = 1000

                # check comparison lengths are the same 
                if len(df_eval[comp][i]) == len(df_eval[x][i]) and comp != x:
                    a = df_eval[comp][i]
                    b = df_eval[x][i]
                    p = stats.ttest_rel(a, b).pvalue
                if x + i not in s:
                    s[x + i] = []
                if p < 0.01:
                    s[x + i].append(symbols[c]*2)
                elif p < 0.05:
                    s[x + i].append(symbols[c])
            cnt+=1

    for x in df:
        for i in df[x].index:
            if x + i in s:
                if df[x][i] <= 1:
                    df2[x][i] = "{:.4f}".format(df[x][i]) + "$^{" + ",".join(s[x + i]) + "}$"
                else:
                    df2[x][i] = "{}".format(int(df[x][i])) + "$^{" + ",".join(s[x + i]) + "}$"
    if metrics == None:
        
        print(df2.T.rename(columns=metrics).to_latex(escape=False))
    else:
        print(df2.T[metrics.keys()].rename(columns=metrics).to_latex(escape=False))
        
    return df2.T


# from operator import itemgetter
CUTOFF = 100

class Interpolater: 
    def __init__(self, base_path: str, cutoff:int=CUTOFF, normalize:bool=True):
        self._bp = base_path
        self._norm = normalize
#         self._base_res = {}
        self._base_res = self._read_and_normalize(self._bp, self._norm)

    def _read_and_normalize(self, path: str, normalize: bool) -> Dict[str, Union[Dict[str, int], List[float]]]:
        queries = {}
        with open(path) as f:
            for line in f: 
                parts = line.split()
                q = int(parts[0])
                vals = queries.get(q, [{}, []])
                vals[0][parts[2]] = len(vals[1])
                vals[1].append(float(parts[4]))
                queries[q] = vals
        
        for k, v in queries.items():
            v[1] = np.array(v[1]) 
            nm = np.linalg.norm(v[1])
            if nm != 0.0:
                v[1] /= nm   
            v[1] = v[1].tolist()
            queries[k] = v
            
        return queries

    def interpolate(self, path: str, _lambda: float, out_path: str, norm: bool = True): 
        """ Interpolate trec result files and write to tmp file """
        
        queries = copy.deepcopy(self._base_res)
        
        inv = 1.0 - _lambda
        
        if _lambda != 0.0: 
            
            if norm:
                q2 = self._read_and_normalize(path, self._norm)
                if _lambda == 1.0: 
                    queries = q2
                else:
                    for k, v in q2.items():
                        vals = queries.get(k, None)
                        for j, ind in v[0].items(): 
                            ind2 = vals[0].get(j, -1)
                            if ind2 == -1:
                                vals[1].append(v[1][ind]*_lambda)
                            else:
                                vals[1][ind2] = inv*vals[1][ind2] + v[1][ind]*_lambda
                                
                        
            else:
                with open(path) as f:
                    for line in f: 
                        parts = line.split()
                        q = int(parts[0])
                        vals = queries.get(q, None)
                        if vals is None:
                            raise
                        ind = vals[0].get(parts[2], -1)
                        if ind == -1:
                            vals[1].append(float(parts[4])*_lambda)
                        else:
                            vals[1][ind] = vals[1][ind]*inv + float(parts[4])*_lambda
                        
                        #print(line)
                        #print(vals)
                        #break

        with open(out_path, 'w') as f:
            for q, q_res in sorted(queries.items(), key=lambda x: x[0]):
                q_res = sorted(zip(q_res[0], q_res[1]), reverse=True)
                cut = len(q_res)
                if cut > CUTOFF:
                    cut = CUTOFF
                for i, res in enumerate(q_res[:cut]):
                    f.write('{0} Q0 {1} {2} {3} t\n'.format(q, res[0], i, res[1]))
                    
                    
### In place cross validation with list of runs 
                                 
def cross_validation(runs, folds, metrics, base_qry):
    _max = []
    max_inds = []
    qry_res = []
    
    for m in metrics:
        qry_res.append(runs[0][m])
        max_inds.append([0]*len(folds))
        _max.append([0.0]*len(folds))
    
    for i, run in enumerate(runs):
        for f, fold in enumerate(folds): 
            filtered = run[run.index.isin(fold[1])]
            for j, m in enumerate(metrics.keys()):
                v = filtered[m].mean()
                if v > _max[j][f]:
                    _max[j][f] = v
                    max_inds[j][f] = i
                    for ind, item in run[run.index.isin(fold[0])][m].items():
                        qry_res[j].loc[ind] = item
    
    _max_res = [0.0] * len(metrics)
    base_qry.sort_index(inplace=True)        
    for i, m in enumerate(metrics):
        qry_res[i].sort_index(inplace=True)
        p = stats.ttest_rel(qry_res[i], base_qry[m]).pvalue

        if p < 0.01:
            _max_res[i] = '{0:.4f}'.format(qry_res[i].mean()) + "$^{**}$"
        elif p < 0.05: 
            _max_res[i] = '{0:.4f}'.format(qry_res[i].mean()) + "$^{*}$"
        else:
            _max_res[i] = '{0:.4f}'.format(qry_res[i].mean())

    return _max_res, max_inds, qry_res


def read_folds(path: str):
    folds = []
    with open(path) as f:
        for line in f:
            parts = line.split('] [') 
            test = list(map(int, parts[1].replace(']', '').split(', ')))
            train = list(map(int, parts[0].replace('[', '').split(', ')))
            folds.append((test, train))
            
    return folds


# bold dataframes 
def bold_max(in_df):
    df = in_df.copy().astype('str')
    maxes = df.select_dtypes(['object']).apply(lambda x: x.str.strip('$^{*\dagger,\cdot}')).astype(float).idxmax(axis=0)
    for m, i in maxes.items():
        idx = df.loc[i, m].find('$')
        if idx != -1:
            df.loc[i, m] = '\textbf{' + df.loc[i, m][:idx] + '}' + df.loc[i, m][idx:]
        else:
            df.loc[i, m] = '\textbf{' + str(df.loc[i, m]) + '}' 
            
    return df

def write_table(path: str, content: str):
    with open(path+'.tex', "wt") as f:
        f.write(content)