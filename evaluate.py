#!/usr/bin/env python

from __future__ import print_function
import sys 
import os
from pdb import set_trace

import sklearn.datasets.svmlight_format as svmlight
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import *

from nltk.metrics import ConfusionMatrix

from cv import XValMT

root_dir = "/home/chase/NLP/Irish/IrishDialects"

def warning(*objs):
        print("evaluate.py: WARNING: ", *objs, file=sys.stderr)

def which_classifier(clfer):
    """Initialize your classifier of choice"""
    if clfer == 'SGD':
        #mean: 0.67203, std: 0.00024, params: {'alpha': 1e-06, 'n_iter': 15000},
        params = {'alpha':0.000001,'n_iter':15000,'C':1,'n_jobs':2}
        return SGDClassifier(**params)
    if clfer == "MNB":
        return MultinomialNB()
    if clfer == 'BNB':
        params = {class_prior:[0.041175856307435255,0.9588241436925647]}
        return BernoulliNB(**params)
    if clfer == 'SVC':
        params = {'cache_size':4000,'n_jobs':2,'C':.1}
        return svm.SVC(**params)
    if clfer == 'DT':
        params = {'max_depth':3}
        return DecisionTreeClassifier(**params)
    else:
        warning("Classifier not recognized: {}".format(clfer))
        sys.exit(1)


def mp(t,clf,k=None):
    trainI,testI = t
    if k:
        ch2 = SelectKBest(chi2, k=k)
        #do chi2 fit on train data
        best = ch2.fit(feat_vecs[trainI], labels[trainI])         
        # test data reduced to same k features
        test_feats = best.transform(feat_vecs[testI]).toarray()
        # train data reduced to same k features
        train_feats = best.transform(feat_vecs[trainI]).toarray()     
    else:
        test_feats = feat_vecs[testI]
        train_feats = feat_vecs[trainI]
    train_labels = labels[trainI]
    test_labels = labels[testI]
    # make NB model on train data
    fitted = clf.fit(train_feats, train_labels)
    pred = fitted.predict(test_feats) # predict labels for test
   #set_trace()
    f1 = f1_score(test_labels, pred,average='weighted')
    pre = precision_score(test_labels, pred)
    rec = recall_score(test_labels, pred)
    prf1 = [pre,rec,f1]
    r = []
    m = confusion_matrix(test_labels, pred)
    r.append(m)
    r.append(ConfusionMatrix(test_labels.tolist(), pred.tolist()))
    r.append(classification_report(test_labels,pred))
    r.append(prf1)
    return r



if __name__ == '__main__':
    if len(sys.argv) > 1: 
        svm_light_in  = sys.argv[1]
    else:
        warning("No feature file loaded")
        sys.exit(1)
    print ("Loading dataset %s..."%svm_light_in)
    feat_vecs,labels = svmlight.load_svmlight_file(svm_light_in)
    print ('done\n')
    
    clf = which_classifier("SGD")
    print("Using {} classifier".format(clf.__class__.__name__))
    q = XValMT(labels,n_folds = 10) 
    q.run(mp)
    
#avg_con_matx = array([[0,0],[0,0]])
#avg_pre = 0.0
#avg_rec = 0.0
#avg_f1 = 0.0
    for n,i in enumerate(q.results):
#    avg_con_matx += i[0]
#    avg_pre += i[3][0]
#    avg_rec += i[3][1]
#    avg_f1 += i[3][2]

        with open(root_dir+"/results/"+
                os.path.basename(os.path.splitext(svm_light_in)[0])
                + ".results",'a') as fo:
            fo.write ('-'*35+'Fold%d'%n+35*'-')
            fo.write(str(i[1]))
            fo.write(str(i[2]))