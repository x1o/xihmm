
# coding: utf-8

# # Hidden Markov Model
# 
# $
# \def\a{\alpha}
# \def\g{\gamma}
# \def\P{\mathsf{P}}
# \def\RR{\mathbb{R}}
# \def\idx#1{\hat{#1}}
# \def\b{\beta}
# \def\range#1#2{\overline{#1:#2}}
# \def\L{\mathcal{L}}
# \def\E{\mathsf{E}}
# \DeclareMathOperator*{\argmax}{arg\,max}
# \def\O{\mathcal{O}}
# $

# ## Global settings

# ### Imports

# In[231]:

import math
from random import random, uniform
from itertools import repeat
from itertools import product as cart_prod
from pprint import pprint, pformat
from functools import reduce, partial
import operator
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ### Plotting

# In[232]:

get_ipython().magic('matplotlib inline')
matplotlib.style.use('ggplot')

plt.rcParams['savefig.dpi'] = 72
plt.rcParams['figure.dpi'] = 72
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12  
plt.rcParams['ytick.labelsize'] = 12 
plt.rcParams['legend.fontsize'] = 12 
plt.rcParams['font.family'] = 'serif'  
plt.rcParams['font.serif'] = ['Computer Modern Roman']  
plt.rcParams['text.usetex'] = True 
plt.rcParams['text.latex.unicode'] = True
# plt.rcParams['figure.figsize'] = 5.2, 3.2   # 0.8
# plt.rcParams['figure.figsize'] = 3.9, 2.4   # 0.6
# plt.rcParams['figure.figsize'] = 2.9, 1.8   # 0.45
plt.rcParams['figure.figsize'] = 4.5, 2.8   # 0.7
plt.rcParams['text.latex.preamble'] = r'\usepackage[utf8]{inputenc}'
plt.rcParams['text.latex.preamble'] = r'\usepackage[russian]{babel}'


# ### Options

# In[233]:

# 5: stacktrace
# 6: calculations detailed to terms
# 7: initial data
# TODO: other
VERBOSITY = 6
# as the second argument to round()
OUT_PREC = 3
# for tree-like output
OUT_INCR = '    '

np.set_printoptions(precision=OUT_PREC, suppress=True)


# ## Auxiliary data structures and functions

# In[234]:

class LabeledList(list):
    def __init__(self,  iterable, labels):
        self.labels = labels
        self.idx_dict = {labels[i] : i for i in range(len(labels))}
        super().__init__(iterable)
    def _label_to_idx_proxy(self, method_name, key, *args):
        # allow raw int indices
        if isinstance(key, int):
            idx = key
        else:
            idx = self.idx_dict[key]
        return getattr(super(), method_name)(idx, *args)
    def __getitem__(self, key):
        return self._label_to_idx_proxy('__getitem__', key)
    def __setitem__(self, key, value):
        return self._label_to_idx_proxy('__setitem__', key, value)
    def __sub__(self, ll):
        res = deepcopy(self)
        for i in range(len(ll)):
            res[i] -= ll[i]
        return res
#     def __add__(self, val):
#         if isinstance(val, int):
#             res = sum(self) + val
#         else:
#             res = deepcopy(self)
#             for i in range(len(ll)):
#                 res[i] += ll[i]
#         return res
    def get_sum(self):
        if isinstance(self[0], LabeledList):
            return sum(ll.get_sum() for ll in self)
        else:
            return sum(self)
    def get_abs(self):
        if isinstance(self[0], LabeledList):
            return LabeledList((ll.get_abs() for ll in self), self.labels)
        else:
            return LabeledList((abs(el) for el in self), self.labels)
    def get_max(self):
        if isinstance(self[0], LabeledList):
            return max(ll.get_max() for ll in self)
        else:
            return max(self)
#     def __repr__(self):
#         # ugly
#         return np.array(self).__repr__()[6:-1].replace('      ', '')


# ### (Tests)

# In[235]:

S = 'HL'
Q = 'ACGT'
B = [[0.2, 0.3, 0.3, 0.2],
     [0.3, 0.2, -0.8, 0.4]]

L1 = LabeledList(B[0], Q)
L2 = LabeledList(B[1], Q)
L = LabeledList([L1, L2], S)
print(L['H']['A'])
print(L)
L['H']['A'] = 0.3
print(L['H']['A'])
# print(sum(L1))
# L1 + 2
print(L1.get_sum())
print(L.get_sum())
print(L1-L2)
print(L1)
print(L-L)
print((L1-L2).get_abs())
print(L.get_abs())
print(L.get_abs().get_max())


# In[236]:

def create_dataframe(M, labels_row, labels_col, op = lambda x: x):
    return LabeledList([LabeledList(map(op, row), labels_col) for row in M],
                       labels_row)

# prod = partial(reduce, lambda x, y: x * y)
def prod(seq):
    return reduce(lambda x, y: x * y, seq, 1)

def dbg(msg, verbosity=5, pp=False, **kwargs):
    if verbosity <= VERBOSITY:
        print_func = pprint if pp else print
        print_func(msg, **kwargs)
        
def verbose_sum(X, out_offset=''):
    result = sum(X)
    if VERBOSITY >= 6:
        print(out_offset + ' + '.join(str(round(x, OUT_PREC)) for x in X) + ' = ' + str(round(result, OUT_PREC)))
    return result


# ## Initialization

# In[237]:

class HMM(object):
    pass


# ### Initialize a noisy stochastic vector
# Let $$v:=c\cdot\left(\frac{1}{n}+\alpha_{i}\right)_{i=0}^{n-1}.$$
#   Then $$\sum_{i=0}^{n-1}v_{i}=1\iff\sum_{i=0}^{n-1}c\cdot\left(\frac{1}{n}+\alpha_{i}\right)=1$$
#   holds iff $$c=\frac{1}{\sum_{i=0}^{n-1}\left(1/n+\alpha_{i}\right)}=\frac{1}{1+\sum_{i=0}^{n-1}\alpha_{i}}$$
#  

# In[238]:

def create_stochastic_vector(n, noise=2):
    base = 1. / n
    eps = 1. / n**noise
    noise = [uniform(-eps, eps) for _ in range(n)]
    norm_fact = 1 / (1 + sum(noise))
    return [(base + noise[i]) * norm_fact for i in range(n)]

def create_stochastic_matrix(m, n, noise=2):
    return [create_stochastic_vector(n) for _ in range(m)]


# ### (Tests)

# In[239]:

v = create_stochastic_vector(10, noise=1)
pprint(v)
print(sum(v))

M = create_stochastic_matrix(2, 3)
M


# In[240]:

def __init__(self,
        S: "hidden state labels",
        Q: "observation labels",
        priors: r'P(\xi_0 = s_i)' = None,
        A: r'P(\xi_t = s_j | \xi_{t-1} = s_i) state transition matrix' = None,
        B: r'P(\eta_t = q_k | \xi_t = s_i) observation matrix' = None,
        noise = 2):
    """Initialize HMM.

    If any of the priors vector, A or B matrices aren't provided,
    they are initialized with a suitable row-stochastic vector / matrix.
    """
    
    self.S = S
    self.Q = Q

    if not priors:
        priors = create_stochastic_vector(len(S), noise=noise)
    self.priors = LabeledList(priors, S)
    
    if not A:
        A = create_stochastic_matrix(len(S), len(S), noise=noise)
    self.A = create_dataframe(A, S, S)
    
    if not B:
        B = create_stochastic_matrix(len(S), len(Q), noise=noise)
    self.B = create_dataframe(B, S, Q)
    
    self.create_log_matrices()

def __repr__(self):
    return reduce(lambda x, y: x + '\n' + y,
                  (pformat(obj) for obj in [self.priors, self.A, self.B]))

def init_mem(self, name, t, dim=1, init_val=0):
    create_ll = lambda elem, labels: LabeledList([deepcopy(elem) for _ in range(len(self.S))], labels)
    self.mem[name] = [reduce(create_ll, [self.S] * dim, init_val)
                      for _ in range(t)]
        
    
def create_log_matrices(self):
    # to be used in viterbi
    log2_safe = lambda x: math.log2(x) if x != 0 else float('-inf')
    self.priors_log = LabeledList(map(log2_safe, self.priors), self.S)
    self.A_log = create_dataframe(self.A, self.S, self.S, log2_safe)
    self.B_log = create_dataframe(self.B, self.S, self.Q, log2_safe)

HMM.__init__ = __init__
HMM.__repr__ = __repr__
HMM.init_mem = init_mem
HMM.mem = {}
HMM.create_log_matrices = create_log_matrices


# ### (Tests)

# In[241]:

S = 'HL'
Q = 'ACGT'

priors = [0.5, 0.5]
A = [[0.5, 0.5],
     [0.4, 0.6]]
B = [[0.2, 0.3, 0.3, 0.2],
     [0.0, 0.2, 0.0, 0.8]]

hmm = HMM(S, Q, priors, A, B)
print(hmm)
print()
print(hmm.priors_log)
print(hmm.B_log)
print()
hmm = HMM(S, Q)
print(hmm)
print()

hmm.init_mem('alpha', 10)
print(hmm.mem['alpha'])

hmm.init_mem('gamma', 10, dim=2)
pprint(hmm.mem['gamma'])
hmm.mem['gamma'][0][0][0] = 1
hmm.mem['gamma'][0][0][1] = 2
pprint(hmm.mem['gamma'])


# ## Observation probabilities
# 
# For a given observation sequence $Y$, find its probability $\P(Y)$.

# ### Naive approach
# 
# $$\begin{eqnarray*}
# \P(Y) & = & \P\left(\bigvee_{X}\left(X\cap Y\right)\right)=\sum_{X}\P(XY)=\sum_{X}\left(\P(Y\mid X)\cdot\P(X)\right)\\
#  & = & \sum_{X}\left(\P(Y_{0}\mid X_{0})\P(X_{0})\prod_{t=1}^{T-1}\P(Y_{t}\mid X_{t})\P(X_{t}\mid X_{t-1})\right).
# \end{eqnarray*}$$
# 
# Complexity: $\O(2TN^T)$    

# In[242]:

def pr_YX(self, Y, X):
    return self.B[X[0]][Y[0]] * self.priors[X[0]] *             prod(self.B[X[t]][Y[t]] * self.A[X[t-1]][X[t]]
                 for t in range(1, len(Y)))

def pr_Y_naive(self, Y):
    return sum(self.pr_YX(Y, X)
       for X in cart_prod(self.S, repeat=len(Y)))

HMM.pr_YX = pr_YX
HMM.pr_Y_naive = pr_Y_naive


# ### Example due to Stamp (2012)

# In[243]:

S = ['H', 'C']          # Annual temperatures: hot, cold
Q = ['S', 'M', 'L']     # Size of tree growth rings: small, medium, large
priors = [0.6, 0.4]
A = [[0.7, 0.3],
     [0.4, 0.6],]
B = [[0.1, 0.4, 0.5],
     [0.7, 0.2, 0.1],]

hmm_stamp = HMM(S, Q, priors, A, B)


# In[244]:

Y = ('S', 'M', 'S', 'L')
list(cart_prod(S, repeat=len(Y)))


# In[246]:

print(hmm_stamp.pr_Y_naive(Y))


# ### Forward algorithm
# 
# $$\P(Y)=\sum_{s\in S}\a_{T-1}(s)$$ where $$\begin{eqnarray*}
# \a_{t}(s) & = & \P(Y_{0}\ldots Y_{t},\xi_{t}=s)\\
#  & = & B_{\idx s\idx y_{t}}\sum_{r\in S}A_{\idx r\idx s}\a_{t-1}(r),\quad t\in\range 1{T-1}\\
# \a_{0}(s) & = & B_{\idx s\idx y_{0}}\pi_{\idx s}.
# \end{eqnarray*}$$
# 
# Complexity: $\O(N^2T)$
# 
# If `with_scaling`, also compute $\mathbf{c}=c_0,\dots,c_{T-1},$ where $$c_{t}:=\frac{1}{\sum_{r\in S}\tilde{\a}_{t}(r)},$$ and scale every $\a_t$ as \begin{eqnarray*}
# \bar{\a}_{t}(s) & := & \tilde{\a}_{t}(s)\cdot c_{t}\\
# \tilde{\a}_{t}(s) & := & B_{\idx s\idx y_{t}}\sum_{r\in S}A_{\idx r\idx s}\bar{\a}_{t-1}(r)
# \end{eqnarray*}

# In[247]:

def alpha(self, Y, with_scaling=False):
    self.init_mem('alpha', len(Y))
    a = self.mem['alpha']
    if with_scaling:
        self.mem['c'] = [0] * len(Y)
        c = self.mem['c']
    # Compute \alpha_0 (s) \forall s \in S
    for s in self.S:
        a[0][s] = self.priors[s] * self.B[s][Y[0]]
        if with_scaling:
            c[0] += a[0][s]
    # Scale alpha_0
    if with_scaling:
        c[0] = 1 / c[0]
        for s in self.S:
            a[0][s] *= c[0]
    # Compute \alpha_t(s) \forall s \in S
    for t in range(1, len(Y)):
        for s in self.S:
            for r in self.S:
                a[t][s] += self.A[r][s] * a[t-1][r]
            a[t][s] *= self.B[s][Y[t]]
            if with_scaling:
                c[t] += a[t][s]
        # Scale \alpha_t
        if with_scaling:
            c[t] = 1 / c[t]
            for s in self.S:
                a[t][s] *= c[t]

HMM.alpha = alpha


# In[248]:

def pr_Y_forward(self, Y, with_scaling=False):
    self.alpha(Y, with_scaling)
    return sum(self.mem['alpha'][-1])

HMM.pr_Y_forward = pr_Y_forward


# In[249]:

def pr_Y(self, Y, algorithm='forward', with_scaling=False):
    if algorithm == 'naive':
        return self.pr_Y_naive(Y)
    elif algorithm == 'forward':
        return self.pr_Y_forward(Y, with_scaling)

HMM.pr_Y = pr_Y


# In[250]:

print(hmm_stamp.pr_Y_forward(Y))


# In[251]:

Y = ('S', 'M', 'S', 'L')
print(hmm_stamp.mem)
print(hmm_stamp.mem['alpha'][0]['H'])


# In[252]:

Y = ('M')
hmm_stamp.alpha(Y)
print(hmm_stamp.mem)


# In[253]:

Y = ('S', 'M', 'S', 'L') 
print(hmm_stamp.pr_Y(Y, 'forward'))
print(hmm_stamp.pr_Y(Y, 'naive'))

Y = ('S') 
print(hmm_stamp.pr_Y(Y, 'forward'))
print(hmm_stamp.pr_Y(Y, 'naive'))


# In[254]:

print("Initialized this HMM: ")
print(hmm_stamp)
print()

for algorithm in ('naive', 'forward'):
    print('Observation probabilities using %s algorithm' % algorithm)
    total_pr = 0
    max_prob = 0
    winner = []
    for Y in cart_prod(Q, repeat=3):
        p = hmm_stamp.pr_Y(Y, algorithm=algorithm)
        if p > max_prob:
            max_prob = p
            winner = Y
        print('%s: %s' % (Y, p))
        total_pr += p

    print("Total probability: %f" % total_pr)
    print("Winner: ", end='')
    print(winner)
    print()


# ## Uncovering hidden states
# 
# For a given observation sequence $Y$, find its corresponding hidden states $X^*$ so that $$\argmax_X \P(X\mid Y) = X^*$$

# ### Brute force
# 
# Compute $\P(XY)$ over all possible state sequencies $X$ given $Y$.  Choose $X$ as the winner such that it maximizes $\P(XY)$.

# In[255]:

def compute_total_prob_table(self, Y):
    prob_table = {}
    for X in cart_prod(self.S, repeat=len(Y)):
        prob_table[''.join(X)] = self.pr_YX(Y, X)
    # exactly pr_Y(Y)
    norm_factor = sum(prob_table.values())
    for key in prob_table:
        prob_table[key] = (prob_table[key], prob_table[key] / norm_factor)
    dbg('Prob. table:')
    dbg(prob_table, pp=True)
    return prob_table

def uncover_bruteforce(self, Y):
    prob_table = self.compute_total_prob_table(Y)
    fmt = ' %%%ss | %%8s | %%10s ' % len(Y)
    print(fmt % ('X', 'P(XY)', 'P(XY)/P(Y)'))
    for X, prob in sorted(prob_table.items(), key=lambda item: item[0]):
            print(" %s | %f | %f " % (X, prob[0], prob[1]))
    opt_X, max_prob = max(prob_table.items(), key=lambda item: item[1][1])
    max_prob = max_prob[1]
    return (opt_X, max_prob)

HMM.compute_total_prob_table = compute_total_prob_table
HMM.uncover_bruteforce = uncover_bruteforce


# In[256]:

Y = ('S', 'M', 'S', 'L')
hmm_stamp.uncover_bruteforce(Y)


# ### Forward-backward demo
# 
# At each step t, choose $x_{t}^{*}\in S$ such that $$x_{t}^{*}=\argmax_{X:\xi_{t}=x_{t}^{*}}\P(XY)$$
#  

# In[257]:

def uncover_fb_demo(self, Y):
    prob_table = self.compute_total_prob_table(Y)
    max_prob_states = [0] * len(Y)
    print('Sum of probabilities of the sequences:')
    for t in range(len(Y)):
        max_prob = 0
        for s in self.S:
            prob = sum(pr[0] for (X, pr) in prob_table.items() if X[t] == s)
            print("%s: %f" % (t * '*' + s + (len(Y)-t-1) * '*', prob))
            if prob > max_prob:
                max_prob = prob
                max_prob_states[t] = s
        print('Max (t = %d): %s' % (t, max_prob_states[t]))
    opt_X = ''.join(max_prob_states)
    max_prob = prob_table[opt_X][1]
    return (opt_X, max_prob)

HMM.uncover_fb_demo = uncover_fb_demo


# In[258]:

hmm_stamp.uncover_fb_demo(Y)


# ### Forward-backward
# 
# Let $$\g_{t}(s)=\P\left(\xi_{t}=s\mid Y\right)=\frac{\P(\xi_{t}=s,Y)}{\P(Y)}=\frac{\a_{t}(s)\b_{t}(s)}{\P(Y)}$$ where $\b_{t}(s)=\P(Y_{t+1}\cdots Y_{T-1}\mid\xi_{t}=s)$ and is computed as
# $$\begin{eqnarray*}
# \b_{t}(s) & = & \sum_{r\in S}\b_{t+1}(r)B_{\idx r\idx y_{t+1}}A_{\idx s\idx r},\quad\forall t\in\range 0{T-2}\\
# \b_{T-1}(s) & = & 1.
# \end{eqnarray*}$$
# 
# Then choose $X^*$ such that $$X^{*}=(x_{0},\ldots,x_{T-1}:x_{t}=\argmax_{s\in S}\gamma_{t}(s),\quad t\in\range 0{T-1}).$$
#  
# 

# In[259]:

def beta(self, Y, scale=False):
    self.init_mem('beta', len(Y));
    b = self.mem['beta']
    if scale:
        c = self.mem['c']
    for s in self.S:
        b[len(Y)-1][s] = 1 if not scale else c[len(Y)-1]
    for t in reversed(range(len(Y)-1)):
        for s in self.S:
            for r in self.S:
                b[t][s] += self.A[s][r] * self.B[r][Y[t+1]] * b[t+1][r]
#             b[t][s] = sum(b[t+1][r] * self.B[r][Y[t+1]] * self.A[s][r]
#                           for r in self.S)
            if scale:
                b[t][s] *= c[t]

HMM.beta = beta


# In[260]:

hmm_stamp.beta(Y)
print(hmm_stamp.mem['beta'])


# In[261]:

def gamma(self, Y):
    self.init_mem('gamma', len(Y))
    self.alpha(Y)
    self.beta(Y)
    pr_Y = sum(self.mem['alpha'][-1])
    for t in range(len(Y)):
        for s in self.S:
            self.mem['gamma'][t][s] = self.mem['alpha'][t][s] * self.mem['beta'][t][s] / pr_Y
            
HMM.gamma = gamma


# In[262]:

hmm_stamp.gamma(Y)
print(hmm_stamp.mem['gamma'])


# In[263]:

def backtrack(self, mem):
    dbg('Backtracking:')
    path = [None] * len(mem)
    for t, step in enumerate(mem):
        dbg('t = %s' % t)
        max_state_prob = -float('inf')
        for i, state_prob in enumerate(step):
            dbg('%s: %s' % (self.S[i], state_prob))
            if state_prob > max_state_prob:
                max_state_prob = state_prob
                path[t] = self.S[i]
    return path

HMM.backtrack = backtrack


# In[264]:

def uncover_forward_backward(self, Y):
    self.gamma(Y)
    path = self.backtrack(self.mem['gamma'])
    opt_X = ''.join(path)
    max_prob = self.pr_YX(Y, opt_X)
    return (opt_X, max_prob)

HMM.uncover_forward_backward = uncover_forward_backward


# In[265]:

hmm_stamp.uncover_forward_backward(Y)


# In[266]:

0.0018816 / hmm_stamp.pr_Y(Y)


# ### Viterbi alogirthm
# Find $$\argmax_{X}\P(X\mid Y)=(x_{0},\ldots,x_{T-1}:x_{t}=\argmax_{s\in S}f_{t}(s),\quad t\in\range 0{T-1})$$ where
# $$\begin{eqnarray*}
# f_{t}(s) & = & \max_{x_{0},\ldots,x_{t-1}\in S}\P(X_{0}\ldots X_{t-1},\xi_{t}=s,Y_{0}\ldots Y_{t})\\
#  & = & B_{\idx s\idx y_{t}}\max_{r\in S}A_{\idx r\idx s}f_{t-1}(r)\\
# f_{0}(s) & = & B_{\idx s\idx y_{0}}\cdot\pi_{\idx s}.
# \end{eqnarray*}$$
# 
# Avoid vanishing probabilities as $t \to \infty$: compute $$\tilde{f}_{t}(s)=\log f_{t}(s)=\max_{r\in S}\left(\log A_{\idx r\idx s}+\log B_{\idx s\idx y_{t}}+\tilde{f}_{t-1}(r)\right).$$

# In[267]:

def viterbi(self, Y, with_log=True, algorithm='iterative'):
    def viterbi_iter(Y, op):
        for s in S:
            mem[0][s] = op(B[s][Y[0]], priors[s])
            dbg('f_%s (%s) = %s' % (0, s, mem[0][s]))
        for t in range(1, len(Y)):
            for s in S:
                mem[t][s] = -float('inf')
                for r in S:
                    cur_val = op(A[r][s], mem[t-1][r])
                    if cur_val > mem[t][s]:
                        mem[t][s] = cur_val
                mem[t][s] = op(mem[t][s], B[s][Y[t]])
                dbg('f_%s (%s) = %s' % (t, s, mem[t][s]))
    
    def viterbi_rec(t, s, op, out_offset=''):
        dbg(out_offset + 'f_%s (%s):' % (t, s))
        if mem[t][s] is not None:
            p = mem[t][s]
        else:
            if t == 0:
                p = op(B[s][Y[t]], priors[s], out_offset=out_offset)
            else:
                p = op(B[s][Y[t]],
                        max(op(A[r][s],
                               viterbi_rec(t-1, r, op, out_offset+OUT_INCR),
                               out_offset=out_offset)
                            for r in S),
                        out_offset=out_offset)
            dbg(out_offset+'Appending state to mem...', 7)
            dbg(out_offset+'%s -> ' % mem, 8, end='')
            mem[t][s] = p
            dbg(out_offset+'%s' % mem, 7)
        dbg(out_offset + '==> f_%s (%s) = %s' % (t, s, round(p, OUT_PREC)), 5)
        return p
    
    if with_log:
        A = self.A_log
        B = self.B_log
        priors = self.priors_log
        op = lambda *args, **kwargs: verbose_sum(args, **kwargs)
    else:
        #TODO: verbose prod
        A = self.A
        B = self.B
        priors = self.priors
        op = lambda *args, **kwargs: reduce(operator.mul, args, 1)
    S = self.S
    self.init_mem('viterbi', len(Y), init_val=None)
    mem = self.mem['viterbi']
    dbg('Using matrices:', 8)
    dbg('A = ', 8)
    dbg(A, 8)
    dbg('B = ', 8)
    dbg(B, 8)
    if algorithm == 'recursive':
        for s in S:
            viterbi_rec(len(Y)-1, s, op, out_offset='')
            print()
    elif algorithm == 'iterative':
        viterbi_iter(Y, op)
    else:
        raise Exception(NotImplemented)
        
    # backtracking
    dbg(mem)
    return self.backtrack(mem)

HMM.viterbi = viterbi


# #### Example due to Borodovsky & Ekisheva (2006), pp 80-81
# 
# (+ presentation HMM: Viterbi alogirhtm - a toy example)

# In[269]:

S = 'HL'
Q = 'ACGT'

priors = [0.5, 0.5]
A = [[0.5, 0.5],
     [0.4, 0.6]]
B = [[0.2, 0.3, 0.3, 0.2],
     [0.3, 0.2, 0.2, 0.3]]

hmm_borod = HMM(S, Q, priors, A, B)

Y = 'GGCACTGAA'
# Y = 'G'

VERBOSITY = 7
print(hmm_borod.viterbi(Y, algorithm='recursive'))


# In[270]:

print(hmm_borod.viterbi(Y, algorithm='iterative'))


# In[271]:

# viterbi probabilities not meaningful?
Y = ('S', 'M', 'S', 'L')
hmm_stamp.viterbi(Y, with_log=False, algorithm='iterative')


# In[272]:

sum([0.2799999999999999, 0.0448, 0.014111999999999998, 0.0028223999999999996])


# In[273]:

def uncover(self, Y, algorithm='viterbi', **kwargs):
    if algorithm == 'bruteforce':
        return self.uncover_bruteforce(Y)
    elif algorithm == 'forward-backward_demo':
        return self.uncover_fb_demo(Y)
    elif algorithm == 'forward-backward':
        return self.uncover_forward_backward(Y)
    elif algorithm == 'viterbi':
        return self.viterbi(Y, kwargs)
    else:
        raise Exception(NotImplemented)

HMM.uncover = uncover


# #### Simulation

# In[275]:

def simulate(self, n):
    X = [np.random.choice(list(self.S), 1, p=self.priors)[0]]
    Y = [np.random.choice(list(self.Q), 1, p=self.B[0])[0]]
    for _ in range(n - 1):
        X.append(np.random.choice(list(self.S), 1, p=self.A[X[-1]])[0])
        Y.append(np.random.choice(list(self.Q), 1, p=self.B[X[-1]])[0])
    return X, Y

HMM.simulate = simulate


# In[276]:

X, Y = hmm_borod.simulate(100)
VERBOSITY = 4
X_est = hmm_borod.viterbi(Y)
print(hmm_borod)
print(sum(np.array(X) == np.array(X_est)) / 100)


# In[277]:

S = 'AB'
Q = 'XYZ'

priors = [0.5, 0.5]
A = [[0.85, 0.15],
     [0.12, 0.88]]
B = [[0.8, 0.1, 0.1],
     [0.0, 0.0, 1]]
hmm_reference = HMM(S, Q, priors, A, B)

X, Y = hmm_reference.simulate(1000)
VERBOSITY = 4
X_est = hmm_reference.viterbi(Y)
print(hmm_reference)
print(sum(np.array(X) == np.array(X_est)) / 1000)


# In[278]:

X, Y = hmm_stamp.simulate(1000)
VERBOSITY = 4
X_est = hmm_stamp.viterbi(Y)
print(hmm_stamp)
print(sum(np.array(X) == np.array(X_est)) / 1000)


# In[279]:

# res = []
# T = 1000
# hmm_init_rand22 = HMM('AB', 'CD', noise=1)
# for t in range(100):
#     X, Y = hmm_init_rand22.simulate(T)
#     X_est = hmm_init_rand22.viterbi(Y)
#     res.append(sum(np.array(X) == np.array(X_est)) / T)
# plt.tight_layout(pad=0.1)
# plt.plot(res)
# plt.show()
# print(np.mean(res))


# ##### Random matrices

# In[208]:

plt.rcParams['figure.figsize'] = 4.5, 2.8   # 0.7

VERBOSITY = 4
res = []
T = 1000
for t in range(100):
    hmm_init_rand = HMM('AB', 'CDEF', noise=3)
    X, Y = hmm_init_rand.simulate(T)
    X_est = hmm_init_rand.viterbi(Y)
    res.append(sum(np.array(X) == np.array(X_est)) / T)

print(hmm_init_rand)

# with plt.style.context(('grayscale')):
plt.plot(res, label=r'Guesses correct')
plt.ylim(0.4, 1)
plt.axhline(np.mean(res), label='Mean', color='#348ABD')
plt.legend()
plt.xlabel('No. trial')
plt.tight_layout(pad=0.1)
# plt.ylabel('bar')
plt.savefig('viterbi-2_4.pdf')
plt.show()
print(np.mean(res))


# ##### Non-random matrix

# In[215]:

res = []
T = 1000
hmm_nonrandom = HMM('AB', 'CDEF', [0.5, 0.5], [[0.9, 0.1], [0.05, 0.95]], [[0.8, 0.1, 0.05, 0.05], [0.0, 0.95, 0.04, 0.01]])
for t in range(100):
#     hmm_init_rand = HMM('AB', 'CDEF', noise=1)
    X, Y = hmm_nonrandom.simulate(T)
    X_est = hmm_nonrandom.viterbi(Y)
    res.append(sum(np.array(X) == np.array(X_est)) / T)
plt.plot(res, label=r'Guesses correct')
plt.ylim(0.5, 1)
plt.axhline(np.mean(res), label='Mean', color='#348ABD')
plt.legend(loc='lower right')
plt.xlabel('No. trial')
plt.tight_layout(pad=0.1)
# plt.ylabel('bar')
plt.savefig('viterbi-2_4-good.pdf')
plt.show()
print(np.mean(res))


# ## Learning

# In[282]:

def baum_welch_scaled_step(self, Y):
    self.init_mem('delta', len(Y)-1, dim=2)
    self.init_mem('gamma', len(Y))
    self.alpha(Y, with_scaling=True)
    self.beta(Y, scale=True)
    a = self.mem['alpha']
    b = self.mem['beta']
    d = self.mem['delta']    # unnormalized
    g = self.mem['gamma']    # unnormalized
    
    dbg('alpha: %s' % a, pp=True)
    dbg('beta: %s' % b, pp=True)
    
    # Calculate \gamma * P(Y) and \delta * P(Y)
    
    for t in range(len(Y)-1):
        # CUT
#         denom = 0
#         for s in self.S:
#             for r in self.S:
#                 denom += self.A[s][r] * self.B[r][Y[t+1]] * a[t][s] * b[t+1][r]
        # CUT
        for s in self.S:
            for r in self.S:
                d[t][s][r] = self.A[s][r] * self.B[r][Y[t+1]] * a[t][s] * b[t+1][r]
#                 d[t][s][r] = self.A[s][r] * self.B[r][Y[t+1]] * a[t][s] * b[t+1][r] / denom
            g[t][s] = sum(d[t][s])
    
    dbg('delta: %s' % d, pp=True)
    
    # Special case of t = T-1
    denom = 0
    for s in self.S:
        g[len(Y)-1][s] = a[len(Y)-1][s] * b[len(Y)-1][s]
        denom += g[len(Y)-1][s]
    for s in self.S:
        g[len(Y)-1][s] /= denom
        
    dbg('gamma: %s' % g, pp=True)
    
    # Re-esitimate priors, A and B
    
    # priors
    for s in self.S:
        self.priors[s] = g[0][s]
        
    dbg('priors: %s' % self.priors, pp=True)
    
    # A
    for s in self.S:
        for r in self.S:
            numer = denom = 0
            for t in range(len(Y)-1):
                numer += d[t][s][r]
                denom += g[t][s]
            self.A[s][r] = numer / denom
    
    dbg('A: %s' % self.A, pp=True)

    # B
    for s in self.S:
        for q in self.Q:
            numer = denom = 0
            for t in range(len(Y)):
                if Y[t] == q:
                    numer += g[t][s]
                denom += g[t][s]
            self.B[s][q] = numer / denom
        
    dbg('B: %s' % self.B, pp=True)

    return sum(math.log(c) for c in self.mem['c'])   # minus

HMM.baum_welch_scaled_step = baum_welch_scaled_step


# In[283]:

HMM.baum_welch_step = HMM.baum_welch_scaled_step


# In[284]:

S = ['H', 'C']          # Annual temperatures: hot, cold
Q = ['S', 'M', 'L']     # Size of tree growth rings: small, medium, large
priors = [0.6, 0.4]
A = [[0.7, 0.3],
     [0.4, 0.6],]
B = [[0.1, 0.4, 0.5],
     [0.7, 0.2, 0.1],]

hmm_stamp = HMM(S, Q, priors, A, B)


# In[285]:

Y = ('S', 'M', 'S', 'L', 'L', 'M', 'M', 'L', 'S')
VERBOSITY = 4
print(math.log(hmm_stamp.pr_Y(Y)))
print(hmm_stamp.baum_welch_step(Y))
# print(hmm_stamp)


# In[286]:

hmm_stamp = HMM(S, Q, priors, A, B)


# In[287]:

def train(self, Y, eps=0, max_iter=float('inf'), metric='P(Y)', graphics=False):
    if metric == 'P(Y)':
        columns = [r'$-\log \mathsf{P}(Y)$']
    elif metric == 'model':
        columns = [r'$\max |\pi^{(t)} - \pi^{(t-1)}|$', r'$\max |A^{(t)} - A^{(t-1)}|$', r'$\max |B^{(t)} - B^{(t-1)}|$']
    hist = pd.DataFrame(None, columns=columns)
    iter_n = 0
    diff = float('inf')    # "statistic"
    P_Y = None
    while ((iter_n < max_iter) and (diff > eps)):
        dbg('iter #%s' % iter_n, 4, end='')
        if metric == 'model':
            priors_old = deepcopy(self.priors)
            A_old = deepcopy(self.A)
            B_old = deepcopy(self.B)
        p_old = self.baum_welch_step(Y)  # return *old* value of P(Y)
        if metric == 'model':
            model_diff = [(self.priors - priors_old).get_abs().get_max(),
                          (self.A - A_old).get_abs().get_max(),
                          (self.B - B_old).get_abs().get_max()]
            hist.loc[len(hist)] = model_diff
            diff = max(model_diff)
        elif metric == 'P(Y)':
            hist.loc[len(hist)] = p_old
            if P_Y is None:
                P_Y = p_old
            else:
                diff = P_Y - p_old
                P_Y = p_old
        else:
            raise NotImplementedError('Unknown metric "%s"' % metric)
        dbg(' %s' % diff, 4)
        iter_n += 1
    self.create_log_matrices()
    return hist

HMM.train = train


# In[288]:

# def plot_df(df, filename=None):
#     for col in df:
#         plt.plot(df[col], label=col)
#     plt.legend()
#     plt.tight_layout(pad=0.1)
#     if filename:
#         plt.savefig(filename)
#     plt.show()


# ### Metrics

# In[154]:

# plt.rcParams['figure.figsize'] = 3.9, 2.4   # 0.6
matplotlib.style.use('ggplot')
Y = ('S', 'M', 'S', 'L', 'L', 'M', 'M', 'L', 'S')
VERBOSITY = 4
hmm_stamp_copy = deepcopy(hmm_stamp)
for n, metric in enumerate(['P(Y)', 'model']):
    hmm_stamp_copy = HMM(S, Q, priors, A, B)
    h = hmm_stamp_copy.train(Y, metric=metric, max_iter=15)
#     plot_df(h, 'metric-%s.pdf' % metric)
#     with plt.style.context(('grayscale')):
#     plt.plot(h)
#     plt.tight_layout(pad=0.1)
#     plt.show()
#     ax = h.plot(figsize=(10, 7), fontsize=16)
#     ax = h.plot(figsize=(plt.rcParams['figure.figsize']))
#     plt.subplot(2, 1, n+1)
#     h.plot(subplots=True)
    ax = h.plot()
#     print(ax)
    ax.legend()
    plt.xlabel('No. iteratrions')
    ax.figure.tight_layout(pad=0.1)
    ax.figure.savefig('metric-%s.pdf' % metric)
#     ax.tick_params(labelsize=16)
#     ax.legend(fontsize=16)
#     ax.figure.dpi = 72
#     ax.figure.savefig('metric-%s.pdf' % metric, dpi=72)


# ### Simulation

# In[155]:

S = 'AB'
Q = 'XYZ'


# In[156]:

priors = [0.5, 0.5]
A = [[0.85, 0.15],
     [0.12, 0.88]]
B = [[0.8, 0.1, 0.1],
     [0.0, 0.0, 1]]


# In[157]:

hmm_reference_borrowed = HMM(S, Q, priors, A, B)


# In[158]:

hmm_reference_rand = HMM(S, Q)


# In[159]:

hmm_reference = deepcopy(hmm_stamp)


# In[160]:

T = 1000
X, Y = hmm_reference.simulate(T)
print(Y[:20])


# In[162]:

VERBOSITY = 4
print('Expected: ')
print(hmm_reference)
print()
# priors = create_stochastic_vector(len(S))
# A = create_stochastic_matrix(len(S), len(S))
# B = create_stochastic_matrix(len(S), len(Q))
# priors = [0.5, 0.5]
# A = [[0.5, 0.5],
#      [0.5, 0.5]]
# B = [[0.3, 0.3, 0.4],
#      [0.2, 0.5, 0.3]]
# for metric in [('P(Y)', 0.001), ('model', 0.0001)]:
for metric in [('P(Y)', 0.0001)]:
#     hmm_train = HMM(S, Q, priors, A, B)
    hmm_train = HMM(hmm_reference.S, hmm_reference.Q, noise=1)
    print('Initial: ')
    print(hmm_train)
    h = hmm_train.train(Y, metric=metric[0], eps=metric[1], max_iter=300)
#     h = hmm_train.train(Y, metric=metric[0], max_iter=150)
    print('Trained: ')
    print(hmm_train)
    print()
    h.plot()


# In[163]:

ax = h.plot()
ax.legend()
plt.xlabel('No. iteratrions')
ax.figure.tight_layout(pad=0.1)
ax.figure.savefig('test-stamp-random.pdf')


# #### Matrix diff

# In[204]:

plt.rcParams['figure.figsize'] = 2.9, 1.8   # 0.45

import colormaps as cmaps

for attr in ('A', 'B'):
    fig, ax = plt.subplots()
    diff = (getattr(hmm_train, attr) - getattr(hmm_reference, attr)).get_abs()
    cax = ax.matshow(diff, cmap=cmaps.viridis, vmin=0, vmax=max(getattr(hmm_reference, attr))[0])
#     plt.matshow(diff, fignum=100, cmap=cmaps.viridis, vmin=0, vmax=max(getattr(hmm_reference, attr))[0])
#     ax.set_title('blah')
    cbar = fig.colorbar(cax)
    plt.tight_layout(pad=0.2)
    plt.savefig('diff-%s.pdf' % attr)
    plt.show()
    print(diff.get_sum())
#     print(sum(diff))

# fig, axes = plt.subplots(nrows=1, ncols=2)
# attrs = ('A', 'B')
# for i in range(2):
#     diff = (getattr(hmm_train, attrs[i]) - getattr(hmm_reference, attrs[i])).get_abs()
#     im = axes.flat[i].matshow(diff, cmap=cmaps.viridis, vmin=0, vmax=max(getattr(hmm_reference, attr))[0])

# plt.tight_layout()
# fig.colorbar(im, ax=axes.ravel().tolist())


# plt.savefig('diff-together.pdf')
plt.show()


# #### State probability matches

# In[165]:

plt.rcParams['figure.figsize'] = 4.5, 2.8   # 0.7
plt.plot([hmm_train.mem['gamma'][t][hmm_train.S[0]] for t in range(T)], 'o', alpha=0.5)
plt.ylim(-0.1, 1.1)
plt.xlabel('Time step $t$')
plt.ylabel(r'$\gamma_t(s_0) = \mathsf{P}(\xi_t = s_0 \mid Y)$')
plt.tight_layout(pad=0.1)
plt.savefig('state-dist.pdf')


# In[183]:

plt.plot([hmm_train.mem['gamma'][t][hmm_train.S[0]] for t in range(T)][:50], 'o', alpha=0.8, label=r'$\lambda_\mathrm{est}$')
plt.plot((np.array(X)[:50] == hmm_train.S[0]), 'o', alpha=0.8, label=r'$\lambda_\mathrm{ref}$')
plt.legend(numpoints=1)
plt.ylim(-0.1, 1.1)
plt.xlabel('Time step $t$')
plt.ylabel(r'$\gamma_t(s_0) = \mathsf{P}(\xi_t = s_0 \mid Y)$')
plt.tight_layout(pad=0.1)
plt.savefig('state-comp.pdf')


# In[190]:

state_0_dist = [hmm_train.mem['gamma'][t][hmm_train.S[0]] for t in range(T)]
sum((np.array(X) == hmm_train.S[0]) ==
    (np.array(state_0_dist) > 0.5)) / T


# In[184]:

hmm_train.create_log_matrices()


# In[188]:

sum(np.array(hmm_train.viterbi(Y)) == np.array(X)) / T


# ### Marvin the Martian reads brown corpus

# In[ ]:

import sys
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
# sys.setrecursionlimit(10**6)

corpus_path = '/home/xio/Projects/СПбГУ/Курсовая-5/brown_corpus/'

Y = []

for fname in ['A01', 'A02', 'A03', 'A04', 'A05']:
    f = open(corpus_path + fname, 'r')
    for line in f:
        Y += [letter.lower() for letter in list(line[15:].strip())
                if letter.isalpha() or letter == ' ']
    f.close()

Y = Y[:50000]
print(Y[:10])

import string
Q = string.ascii_lowercase+' '
S = ['V', 'C']      # vowel, consonant


# In[ ]:

priors = [0.51316, 0.48684]
A = [[0.47468, 0.52532],
     [0.51656, 0.48344]]

B = [[0.03735,0.03408,0.03455,0.03828,0.03782,0.03922,0.03688,0.03408,0.03875,0.04062,0.03735,0.03968,0.03548,0.03735,0.04062,0.03595,0.03641,0.03408,0.04062,0.03548,0.03922,0.04062,0.03455,0.03595,0.03408,0.03408,0.03688],
     [0.03909,0.03537,0.03537,0.03909,0.03583,0.03630,0.04048,0.03537,0.03816,0.03909,0.03490,0.03723,0.03537,0.03909,0.03397,0.03397,0.03816,0.03676,0.04048,0.03443,0.03537,0.03955,0.03816,0.03723,0.03769,0.03955,0.03397]]

hmm_marvin = HMM(S, Q, priors, A, B)

VERBOSITY = 4

# print("Initialized this HMM: ")
# print(hmm_marvin)


# In[ ]:

print("Training the model:")
get_ipython().magic('time h = hmm_marvin.baum_welch_step(Y)')


# In[ ]:

# for metric in ['P(Y)', 'model']:
#     hmm_marvin = HMM(S, Q, priors, A, B)
#     h = hmm_marvin.train(Y, metric=metric, max_iter=100)
#     h.plot()
hmm_marvin = HMM(S, Q, priors, A, B)
h = hmm_marvin.train(Y, metric='P(Y)', max_iter=100)
h.plot()


# In[ ]:

df = pd.DataFrame(list(zip(*hmm_marvin.B)), columns=['V', 'C'], index=list(hmm_marvin.Q))
ax = df.plot(kind='bar', stacked=True, figsize=(10, 8), fontsize=16)
ax.tick_params(labelsize=14)


# In[ ]:

ax.get_figure().savefig('a.svg')


# In[ ]:

''.join(hmm_marvin.simulate(100)[1])


# In[ ]:

s1 = ''.join(hmm_marvin.simulate(100)[1])
s2 = ''.join(hmm_marvin.simulate(100)[1])
print('%f, %f' % (math.log(hmm_marvin.pr_Y(s1)), math.log(hmm_marvin.pr_Y(s2))))


# In[ ]:

random_s = ''.join(np.random.choice(list(string.ascii_lowercase + ' '), size=100))
math.log(hmm_marvin.pr_Y(random_s))


# ### Marvin the Martian reads Russian news

# In[167]:

import sys
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
# sys.setrecursionlimit(10**6)

corpus_path = '/home/xio/Projects/СПбГУ/Курсовая-5/ru_corpus/'

Q = 'абвгдежзийклмнопрстуфхцчшщъыьэюя '
S = ['C', 'V']      # vowel, consonant

Y = []
with open(corpus_path + 'news.2008.ru.shuffled', 'r') as f:
    if len(Y) < 5 * 10**4:
        for line in f:
            Y += [letter.lower() for letter in list(line.strip())
                  if letter.lower() in Q]
            Y += ' '

Y = Y[:50000]
''.join(Y[:100])


# In[169]:

hmm_marvin_ru = HMM(S, Q)

VERBOSITY = 4

# print("Initialized this HMM: ")
# print(hmm_marvin_ru)


# In[170]:

print("Training the model:")
get_ipython().magic('time h_ru = hmm_marvin_ru.baum_welch_step(Y)')


# In[171]:

hmm_marvin_ru = HMM(S, Q)
h_ru = hmm_marvin_ru.train(Y, metric='P(Y)', eps=0.0001, max_iter=150)
h_ru.plot()


# In[175]:

# df = pd.DataFrame(list(zip(*hmm_marvin_ru2.B)), columns=(hmm_marvin_ru2.S), index=list(hmm_marvin_ru2.Q))
df = pd.DataFrame(list(zip(*hmm_marvin_ru.B)), columns=(['C', 'V']), index=list(hmm_marvin_ru.Q))
ax = df.plot(kind='bar', stacked=True)
ax.figure.tight_layout(pad=0.1)
ax.figure.savefig('marvin_ru.pdf')


# In[ ]:

''.join(hmm_marvin_ru.simulate(100)[1])

