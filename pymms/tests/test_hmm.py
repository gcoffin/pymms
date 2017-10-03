import unittest
import numpy as np
import six
from pymms.hmm import GaussianHMM, DiscreteHiddenMarkov, ClassifierHMM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from hmmlearn import hmm as hmmlearn
import scipy.linalg

class TestEmitDecode(unittest.TestCase):
    def testGMM(self):
        mm = GaussianHMM([[ 0.8, 0.2, 0],
                          [ 0,   0.8, 0.2],
                          [ 0,   0,   1]],
                         [[0,0],[0,1],[1,1]],
                         1
                         )
        np.random.seed(42)
        X = mm.generate_obs()
        s = mm.predict(X)
        self.assertTrue(np.allclose(s,sorted(s)))
    
    def test1(self):
        mm = DiscreteHiddenMarkov(dict(enumerate('abc')),
                                  [[ 0.8, 0.2, 0],
                                   [ 0,   0.8, 0.2],
                                   [ 0,   0,   1]],
                                  [[1,0,0],[0,1,0],[0,0,1]]
                                  )
        self.assertTrue(np.allclose(mm.decode('aaabbc'),[0,0,0,1,1,2]))

    def test_hmmlearn(self):
        params = dict(startprob_ = np.array([0.6, 0.3, 0.1]),
                      transmat_ = np.array([[0.7, 0.2, 0.1],
                                            [0.3, 0.5, 0.2],
                                            [0.3, 0.3, 0.4]]),
                      means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]]),
                      covars_ = np.tile(np.identity(2), (3, 1, 1)))
        model1 = hmmlearn.GaussianHMM(n_components=3, covariance_type="full")
        for k,v in six.iteritems(params):
            setattr(model1,k,v)
        model2 = GaussianHMM(**params)
        states = model2.generate_obs(stop=lambda states:len(states)==100)
        self.assertTrue(np.allclose(model1.predict(states),model2.predict(states)))
        qda = QuadraticDiscriminantAnalysis(store_covariances=True)
        qda.covariances_ = params['covars_']
        qda.means_ = params['means_']
        qda.priors_ = params['startprob_']
        qda.classes_ = np.arange(0,3,dtype=int)
        rotations = []
        scalings = []
        for k in qda.classes_:
            evals,evects = scipy.linalg.eigh( qda.covariances_[k,:,:])
            rotations.append(evects)
            scalings.append(np.sqrt(evals))
        qda.rotations_ = np.array(rotations)
        qda.scalings_ = np.array(scalings)
        model3 = ClassifierHMM(qda,transmat_=params['transmat_'],
                               startprob_ = params['startprob_'])
        self.assertTrue((model1.predict(states) == model3.predict(states)).sum() > len(states) * 0.95)

    def test_fitseq(self):
        params = dict(startprob_ = np.array([0.6, 0.3, 0.1]),
                      transmat_ = np.array([[0.7, 0.2, 0.1],
                                            [0.3, 0.5, 0.2],
                                            [0.3, 0.3, 0.4]]),
                      means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]]),
                      covars_ = np.tile(np.identity(2), (3, 1, 1)))
        model = GaussianHMM(**params)
        states = model.generate_state_seq(stop=lambda states:len(states)==1000)
        obs = model.generate_obs(states)
        model2 = ClassifierHMM(clf=QuadraticDiscriminantAnalysis(store_covariances=True))
        model2.fit_seq(obs,states,[len(states)])
        p = model2.predict(obs)
        self.assertTrue(np.sum(p == states) / len(states) > 0.9)


    def test_jumpall(self):
        params = dict(startprob_ = np.array([0.6, 0.3, 0.1]),
                      transmat_ = np.array([[0.7, 0.2, 0.1],
                                            [0.3, 0.5, 0.2],
                                            [0.3, 0.3, 0.4]]),
                      means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]]),
                      covars_ = np.tile(np.identity(2), (3, 1, 1)))
        model = GaussianHMM(**params)
        states = model.generate_state_seq(stop=lambda states:len(states)==1000)
        obs = model.generate_obs(states)
        model2 = ClassifierHMM(clf=QuadraticDiscriminantAnalysis(store_covariances=True), rand_jump_=0.999999999)
        model2.fit_seq(obs,states,[len(states)])
        self.assertTrue(np.allclose(model2.transmat_, np.ones((3,3))/3.))

    def test_equiprob(self):
        model = GaussianHMM(startprob_ = np.array([0.6, 0.3, 0.1]),
                            transmat_ = np.array([[0.7, 0.2, 0.1],
                                                  [0.3, 0.5, 0.2],
                                                  [0.3, 0.3, 0.4]]),
                            means_ = np.array([[0.0, 0.0], [3.0, -3.0], [5.0, 10.0]]),
                            covars_ = np.tile(np.identity(2), (3, 1, 1)))
        y = model.generate_state_seq(stop=lambda states:len(states)==1000)
        X = model.generate_obs(y)
        
        svm = SVC(probability=True)
        svm.fit(X,y)
        y1 = svm.predict(X)

        cl = ClassifierHMM(startprob_ = np.ones(3,dtype=float)/3.,
                           transmat_ = np.ones((3,3),dtype=float)/3.,
                           clf = svm)
        y2 = cl.predict(X)
        self.assertTrue(list(y1),list(y2))
