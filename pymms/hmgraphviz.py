import graphviz
import numpy as np

class IdDct(object):
    def __getitem__(self,x):
        return x

iddct = IdDct()

def draw(hm,dct=None,threshold=0.02):
    print(np)
    threshold = np.log(threshold)
    g = graphviz.Digraph()
    transitions = hm.transmat_
    dic = getattr(hm,'dic_',dct or iddct)
    for i in range(hm.nstates):
        for j in range(hm.nstates):
            t = np.log(transitions[i,j])
            if t>threshold:
                print(repr(dic[i]),dic[j],t,threshold)
                g.edge(str(dic[i]),str(dic[j]),
                       penwidth=str(int(t - threshold)+0.5),
                       label="%0.2f"%transitions[i,j]
                )
    return g
