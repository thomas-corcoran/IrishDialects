#!/usr/bin/env python

from __future__ import print_function
from pdb import set_trace
import sklearn.datasets.svmlight_format as svmlight
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm 
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn.metrics import *
from sklearn.cross_validation import KFold,StratifiedKFold
from operator import itemgetter
from random import shuffle
from numpy import array
from nltk.metrics import ConfusionMatrix
import sys 
import os
from cv import XValMT

root_dir = "/home/thomas/Irish"
def warning(*objs):
        print("evaluate.py: WARNING: ", *objs, file=sys.stderr)

if len(sys.argv) > 1: 
    svm_light_in  = sys.argv[1]
else:
    warning("No feature file loaded")
    sys.exit(1)

print ("Loading dataset %s..."%svm_light_in)
feat_vecs,labels = svmlight.load_svmlight_file(svm_light_in)
print ('done\n')


# initialize NB classifier
#clf = MultinomialNB()

#clf = BernoulliNB()
#clf.class_prior =[0.041175856307435255,0.9588241436925647]

#clf = svm.SVC()
#clf.cache_size = 4000
#clf.n_jobs = -1
#clf.C = .1

clf = SGDClassifier()
clf.n_jobs = -1
clf.C =1
clf.alpha = 0.000001
clf.n_iter = 15000
#mean: 0.67203, std: 0.00024, params: {'alpha': 1e-06, 'n_iter': 15000},


#clf = DecisionTreeClassifier()
#clf.max_depth = 3

#scores = cross_validation.cross_val_score(clf, feat_vecs, labels, cv=10,scoring='recall')
#print scores
#set_trace()

def mp(t,k=None):
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
    f1 = f1_score(test_labels, pred)
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


q = XValMT(labels,n_folds = 10, indices=True)
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

"""
avg_con_matx = avg_con_matx/10.0
avg_pre /= 10
avg_rec /= 10
avg_f1 /= 10
with open(root_dir+"/results/"+
                os.path.basename(os.path.splitext(svm_light_in)[0])
                + ".results.AVG",'w') as fo:
    fo.write(os.path.basename(os.path.splitext(svm_light_in)[0])+"\n\n")
    fo.write("                        -1      1\n\
                -1  %6.2f %6.2f\n\
                 1  %6.2f  %6.2f\n\n\
                (row = reference; col = test)\n\n"%
                (avg_con_matx[0][0],avg_con_matx[0][1],
                 avg_con_matx[1][    0],avg_con_matx[1][1]))
    fo.write("avg precision: %1.6f\navg recall: %1.6f\navg f1: %1.6f\n"%
                (avg_pre,avg_rec,avg_f1))
"""
"""
kf = StratifiedKFold(labels, n_folds = 10, indices=True)
for n,i in enumerate(kf):
    res = mp(i)
    with open(root_dir+"/results/"+
                os.path.basename(os.path.splitext(svm_light_in)[0])
                + ".results",'w') as fo:
        fo.write('-'*35+'Fold%d'%n+35*'-')
        fo.write(str(res[0]))
        fo.write(str(res[1]))
"""
        
'''
kf = StratifiedKFold(labels, n_folds = 10, indices=True)
for trainI,testI in kf:
#    ch2 = SelectKBest(chi2, k=1000)
#    best = ch2.fit(feat_vecs[trainI], labels[trainI]) #do chi2 fit on train data
#    test_feats = best.transform(feat_vecs[testI]).toarray()   # test data reduced to same k features
#    train_feats = best.transform(feat_vecs[trainI]).toarray() # train data reduced to same k features
    test_feats = feat_vecs[testI]
    train_feats = feat_vecs[trainI]
    train_labels = labels[trainI]
    test_labels = labels[testI]
   # train_feats = feat_vecs[trainI]
   # test_feats = feat_vecs[testI]
    print "Fitting data..."
    fitted = clf.fit(train_feats, train_labels) # make NB model on train data
    print "Making Predictions..."
    pred = fitted.predict(test_feats) # predict labels for test
    #set_trace()
    print ConfusionMatrix(test_labels.tolist(), pred.tolist())
    print classification_report(test_labels,pred)
'''
