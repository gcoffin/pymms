import numbers
import numpy as np
import six
from scipy import stats



def _draw(cumprob):
    # from a cumulative probability vector, extract a random index
    rand = np.random.random()
    return np.searchsorted(cumprob, rand)

class HiddenMarkovModel(object):
    """An abstract class for hidden Markov models."""
    def __init__(self, transmat_, startprob_ = None, rand_jump_=1e-5, priors_weight=0.1, store_backtrack=False):
        """
        :param array(num_states,num_sates) transmat_: matrix a_ij probability of transition from state i to j

        :param array(num_states) startprob_: initial probability of states
        """
        self.nstates = -1
        self.startprob_ = None
        self.transmat_ = None
        self.store_backtrack = store_backtrack
        self.set_params(transmat_=transmat_,startprob_=startprob_,rand_jump_=rand_jump_,priors_weight=priors_weight)

    def set_params(self,transmat_ = None, startprob_ = None, store_backtrack = None, rand_jump_=None, priors_weight=None):
        if transmat_ is not None:
            self.transmat_ = np.asarray(transmat_,dtype = float)
            self.nstates,nstates = self.transmat_.shape
            assert self.nstates == nstates
        if startprob_ is None and self.startprob_ is None and self.nstates>0:
            startprob_ = [1.]+[0.]*(self.nstates - 1)
        if startprob_ is not None:
            self.startprob_ = np.asarray(startprob_,dtype = float)
            assert self.startprob_.shape == (self.nstates,)
        if rand_jump_ is not None:
            self.rand_jump_ = rand_jump_
        if priors_weight is not None:
            self.priors_weight_ = priors_weight
        if store_backtrack is not None:
            self.store_backtrack = store_backtrack

    def get_params(self):
        return dict(transmat_ = self.transmat_,
                    startprob_ = self.startprob_)

    @property
    def cumtransition(self): #todo:cache
        return self.transmat_.cumsum(axis = 1)
        #assert np.all(np.abs(cumtransition[:,-1] - 1) < 1e-5)

    
    def from_observation(self,observation):
        """a function that for an observation o returns the list P(o|state) for all states"""
        raise NotImplementedError()
    
    def log_from_observation(self,obs):
        """must return an array containing log of the P(o|state) for all states"""
        return np.log(self.from_observation(obs))

    def draw_next_state(self,state,rand = None):
        return 
    
    def generate_state_seq(self,stop = None):
        """generate a possible state sequence. Stop is the stop
        condition, by default reaching the last state"""
        start = _draw(np.cumsum(self.startprob_))
        if stop is None:
            stop = lambda x:x[-1]==self.transmat_.shape[0] - 1
        elif isinstance(stop,numbers.Number):
            stop = lambda x,s=stop:len(x)==s
        states = [start]
        while True:
            states.append(_draw(self.cumtransition[states[-1]]))
            if stop(states):
                return states
        
    def likelihood(self,obs,states=None):
        if states is None:
            states = self.predict(obs)
        likelihood = np.product([self.transmat_[s1,s2] for s1,s2 in zip(states[:-1],states[1:])]) 
        return likelihood * np.product(self.from_observation(obs)[state] for obs,state in zip(obs,states))


    def fit_seq(self,X,y,lengths):
        if len(lengths)==len(y):
            starts, = np.nonzero(np.diff(lengths))
        else:
            starts = np.cumsum(lengths)
        states = np.unique(y)
        self.nstates = num_states = np.max(states) + 1
        transmat_ = np.zeros((num_states,num_states),dtype=np.float64)
        first = np.zeros(num_states,dtype = np.uint)
        s=0
        for e in starts:
            Xs = X[s:e]
            ys = y[s:e]
            if len(Xs)==0:
                continue
            for y1, y2  in zip(ys[0:-1],ys[1:]):
                transmat_[y1, y2]+=1.
            first[ys[0]] += 1
            s = e
        transmat_ = transmat_ / transmat_.sum(axis=1).reshape((-1,1))
        self.transmat_ = transmat_ * (1 - self.rand_jump_) + self.rand_jump_/transmat_.shape[0]
        self.startprob_ = (self.priors_weight_ * np.bincount(y) / len(y)
                    + (1 - self.priors_weight_) * first / np.sum(first))
        self.clf_.fit(X,y)

    def predict(self,observations):
        """implementation of Viterbi algorithm"""
        # number of observations
        nobs = len(observations)
        range_states = list(range(self.nstates))
        # will contain best path
        backtrack = np.zeros((nobs,self.nstates),dtype=int)
        q = np.zeros((nobs,),dtype = int)
        pi = np.log(self.startprob_)
        a = np.log(self.transmat_)
        # first observation
        delta = self.log_from_observation(observations[0]) + pi
        for t in range(nobs-1):
            # all elements below are vectors nstate states
            # for each state, probabilty to come from n states
            p = a.T+delta
            # best path for each state
            bt = backtrack[t,:] = np.nanargmax(p,axis=1)
            # probability according to observation
            delta =  self.log_from_observation(observations[t+1]) + p[[range_states,bt]]
            #print(' mx',np.exp(delta))
        # and backtrack
        q[-1]= np.nanargmax(delta)
        for t in range(nobs-1)[::-1]:
            q[t] = backtrack[t,q[t+1]]
        self.log_prob_ = delta[-1]
        if self.store_backtrack:
            self.backtrack_ = backtrack
        return q

class DiscreteHiddenMarkov(HiddenMarkovModel):
    """a specific implementatation when the observations are discrete and we can
    write the matrix B (p(observation|state))"""
    def __init__(self,dic,transmat_,obs_matrix = None,startprob_ = None, store_backtrack=False):
        """:param dict dic: dictionary from obserbed symbol to the sequence number used (line and column index in the matrices
        :param array(num_states,num_states) transmat_: p(St+1 = j|St = i)
        :param array(num_obs, num_states) obs_matrix: p(ot=i|St=j)
        """
        self.obs_matrix_ = None
        self.dic_ = {}
        super(DiscreteHiddenMarkov,self).__init__(transmat_,startprob_)
        self.set_params(obs_matrix=obs_matrix,dic=dic,store_backtrack=store_backtrack)

    def set_params(self,**kwargs):
        obs_matrix = kwargs.pop('obs_matrix',None)
        dic = kwargs.pop('dic',None)
        super(DiscreteHiddenMarkov,self).set_params(**kwargs)
        if dic is not None:
            if isinstance(dic,tuple):
                dic = dict(enumerate(dic))
            self.dic_ = dic # num -> symbol
            self.rev_dic_ = {v:k for k,v in dic.items()}
            assert sorted(dic.keys()) == list(range(len(dic)))
        if obs_matrix is None and self.obs_matrix_ is None: 
            # state is observed directly
            obs_matrix = np.identity(len(self.dic_),dtype = float)
        if obs_matrix is not None:
            self.obs_matrix_ = np.asarray(obs_matrix,dtype = float)
            self.log_obs_matrix_ = np.log(self.obs_matrix_)
            self.rep_obs = np.cumsum(self.obs_matrix_,axis=0)

    def get_params(self):
        result = super(DiscreteHiddenMarkov,self).get_params()
        result['obs_matrix'] = self.obs_matrix_
        result['dic'] = self.dic_
        return result

    def from_observation(self,observation):
        return self.obs_matrix_[observation]

    def log_from_observation(self,observation):
        return self.log_obs_matrix_[observation]
        
    def generate_obs(self,states = None, stop =None):
        result = []
        if states is None:
            states = self.generate_state_seq(stop)
        for s in states:
            i = _draw(self.rep_obs[:,s])
            result.append(self.dic_[i])
        return result

    def decode(self,observed_symbols):
        return self.predict([self.rev_dic_[i] for i in observed_symbols])

class NoLogClassifierHMM(HiddenMarkovModel):
    def __init__(self, clf=None, transmat_=None, startprob_=None,**kwargs):
        """clf is a trained classifier with a method method
        predict_proba or predict_log_proba"""
        self.clf_ = clf
        super(NoLogClassifierHMM,self).__init__(transmat_,startprob_,**kwargs)

    def set_params(self, **kwargs):
        clf_params = {k:v for k,v in six.iteritems(kwargs) if k.startswith('clf__')}
        base_params = {k:v for k,v in six.iteritems(kwargs) if not k.startswith('clf__')}
        self.clf_.set_params(**clf_params)
        super(NoLogClassifierHMM,self).set_params(**base_params)

    def from_observation(self,observation):
        return self.clf_.predict_proba(observation.reshape(1,-1))

class ClassifierHMM(NoLogClassifierHMM):

    def log_from_observation(self,observation):
        return self.clf_.predict_log_proba(observation.reshape(1,-1))

class GaussianHMM(HiddenMarkovModel):
    """use ClassifierHMM with sklearn.discriminant_analysis.QDA instead"""
    def __init__(self,transmat_=None,means_=None,covars_=None,startprob_=None,**kwargs):
        self.means_ = None
        self.covars_ = None
        super(GaussianHMM,self).__init__(transmat_,startprob_)
        self.set_params(means_=means_,covars_=covars_,**kwargs)

    def set_params(self,**kwargs):
        means_ = kwargs.pop('means_',None)
        covars_ = kwargs.pop('covars_',None)
        super(GaussianHMM,self).set_params(**kwargs)
        if means_ is not None:
            means_ = np.array(means_)
            assert means_.shape[0] == self.nstates,'transition, means_ and covars_ need to have the same first dimension'
            _,self.space_dim = means_.shape
            self.means_ = means_
        if covars_ is None and (self.covars_ is None and self.means_ is not None):
            covars_ = 1.
        if covars_ is not None:
            if isinstance(covars_,numbers.Number):
                covars_ = np.ones((self.nstates,),dtype=np.float)*covars_
            else:
                covars_ = np.array(covars_,dtype=np.float)
            assert covars_.shape[0] == self.nstates
            self.covars_ = covars_
        if covars_ is not None or means_ is not None:
            self.pdfs = [stats.multivariate_normal(mu,sigma) for mu,sigma in zip(self.means_,self.covars_)]

    def get_params(self):
        result = super(DiscreteHiddenMarkov,self).get_params()
        result['means_'] = self.means_
        resuls['covars_'] = self.covars_

    def log_from_observation(self,obs):
        return np.fromiter((pdf.logpdf(obs) for pdf in self.pdfs),dtype = np.float,count = len(self.pdfs))

    def from_observation(self,obs):
        return np.exp(self.log_from_observation(obs))

    def generate_obs(self,states = None, stop = None):
        if states is None:
            states = self.generate_state_seq(stop)
        return [self.pdfs[s].rvs() for s in states]

