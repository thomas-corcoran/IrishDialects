#!/usr/bin/python


from pdb import set_trace
import sklearn.datasets.svmlight_format as svmlight
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
from sklearn.cross_validation import KFold
from operator import itemgetter
from random import shuffle
from numpy import array
import sys

print "Loading dataset ..."
svm_light_in = 'featureFile.dat'

feat_vecs,labels = svmlight.load_svmlight_file(svm_light_in)
print 'done\n'

# initialize chi2 filtering object
yes_chi2 = False
k = 20000
ch2 = SelectKBest(chi2, k=k)

# initialize NB classifier
clf = MultinomialNB()
#clf = svm.SVC()

# how to divide sample sizes
bins = 6
samp_size = len(labels)/bins

def createSamples():
    # create dict with indexes with each sample
    sample_dict = {}
    all_posts_indexes = range(len(labels))
    shuffle(all_posts_indexes)
    for i in range(bins):
        sample_dict['sample'+str(i)] = []
        for n in range(samp_size):
            post_number = all_posts_indexes.pop()
            sample_dict['sample'+str(i)].append(post_number)
    return sample_dict    

def testSamples(num_sample_sets):
    """ 
    Runs series of trials, increasing the ratio of training data to testing data
    by keeping amount of testing data constant and increasing total amount of
    training data. 
    number_sample_sets: how many sample sets to combine for training data
    (test data will always only contain one sample that's not in traing data)
    """
    sample_dict=createSamples()
    f1 = []
    accuracy = []
    precision = []
    recall = []
    for j in range(bins):
        train_samples = range(bins)*(num_sample_sets+1)
        
        train_labelsI = []
        train_featsI = []
#        set_trace()
        # put sample sets into one place
        for i in range(j,j+num_sample_sets):
            i = train_samples[i]
            train_labelsI.extend(sample_dict['sample'+str(i)])
            train_featsI.extend(sample_dict['sample'+str(i)])
        if not train_featsI:
            set_trace()
        if yes_chi2:
            # get chi2 feature selection
            best = ch2.fit(feat_vecs[train_featsI], labels[train_labelsI])
            # apply chi2 feature selection to training vector
            train_feat_vec = best.transform(feat_vecs[train_featsI])
        else:
            train_feat_vec = feat_vecs[train_featsI]
        for k in range(bins):
#            set_trace()
            if k not in [train_samples[I] for I in range(j,j+num_sample_sets)]:
                test_labelsI = sample_dict['sample'+str(k)]
                test_featsI = sample_dict['sample'+str(k)]
                if yes_chi2:
                    # apply chi2 feature selection to testing vector
                    test_feat_vec = best.transform(feat_vecs[test_featsI])
                else:
                    test_feat_vec = feat_vecs[test_featsI]
    #            set_trace()
                # create model
                fitted = clf.fit(train_feat_vec, labels[train_labelsI])
                # apply model
                pred = fitted.predict(test_feat_vec)
                # comute scores
                f1.append(f1_score(labels[test_labelsI],pred,average='weighted'))
                accuracy.append(accuracy_score(labels[test_labelsI],pred,average='weighted'))
                precision.append(precision_score(labels[test_labelsI],pred,average='weighted'))
                recall.append(recall_score(labels[test_labelsI],pred,average='weighted'))
#                print "%.4f sample%d"%(results[-1], j)
    return [array(score).mean() for score in (f1,accuracy,precision,recall)]
def shuffleTestSamples(number_of_shuffles):
    """
    Calls testSamples and runs a sample size number_of_sample times
    shuffling the entire data set each time
    """
    results = []
    print "k =" + str(k)
    for w in range(1,bins):
        to_avg= []   
        print "# of samples: --->"+ str(w*samp_size)
        for _ in range(number_of_shuffles):
            to_avg.append(testSamples(w))
        results.append([array(score).mean() for score in to_avg])
    print "Avg F1:"
    print [f[0] for f in results]
    print "Avg Accuracy:"
    print [a[0] for a in results]
    print 'Avg precision:'
    print [p[2] for p in results]
    print 'Avg recall:'
    print [R[3] for R in results]
    print "Sample sizes:"
    print [w*samp_size for w in range(1,bins)]


def xval(num_folds,max_k=feat_vecs.shape[1],num_steps=25,get_best=True,k=None):
    
    def fitAndPredict(trainI,testI,k):
        """ do one round of fitting to chi2, making model, and doing predictions
        on test"""
        ch2 = SelectKBest(chi2, k=k)
        best = ch2.fit(feat_vecs[trainI], labels[trainI]) #do chi2 fit on train data
        test_feats = best.transform(feat_vecs[testI])   # test data reduced to same k features
        train_feats = best.transform(feat_vecs[trainI]) # train data reduced to same k features
       
        train_labels = labels[trainI] # labels for this sample section

        fitted = clf.fit(train_feats, train_labels) # make NB model on train data
        pred = fitted.predict(test_feats) # predict labels for test
        return pred 

    def iterK():
        """ Search through all k, return prediction of model with
        higest F1"""
        f1_k = []
        print "fold--------------> " + str(i)
        for k in range(1,max_k,step_size):
            pred = fitAndPredict(trainI,testI,k)
            
            if k != 1: 
                f1_k.append((f1_score(test_labels, pred,average='weighted'), k))
                print "%d features: f1=%.4f"%(f1_k[-1][1], f1_k[-1][0])
#        set_trace()
        k = sorted(f1_k, key=itemgetter(0))[-1][1] # take highest F1
        pred = fitAndPredict(trainI,testI,k)
        # compute scores
        k_list.append(k)
        return pred
    
    step_size = max_k/num_steps
    kf = KFold(len(labels), num_folds)# indices=True) # make folds   
    f1 = []
    accuracy = []
    precision = []
    recall = []
    k_list = []
    i = 1
    for trainI,testI in kf: 
        test_labels = labels[testI]
        if get_best: # search though k's for best results
            pred = iterK()
        else: # just do one model for k
            pred = fitAndPredict(trainI,testI,k)
        
        f1.append(f1_score(test_labels,pred,average='weighted'))
        accuracy.append(accuracy_score(test_labels,pred))
        precision.append(precision_score(test_labels,pred,average='weighted'))
        recall.append(recall_score(test_labels,pred,average='weighted'))
        i+=1
    f1 = array(f1)
    accuracy = array(accuracy)
    precision = array(precision)
    recall = array(recall)
    k_list = array(k_list)
    print 'Avg F1: ' + str(f1.mean())
    print 'Avg Accuracy: ' + str(accuracy.mean())
    print 'Avg Precision: ' + str(precision.mean())
    print 'Avg Recall: ' + str(recall.mean() )
    print 'Avg k: ' + str(k_list.mean())
    print f1
    print accuracy
    print precision
    print recall
    print k_list
    return [score.mean() for score in (f1,accuracy,precision,recall)]

def searchK(folds,num_steps,max_k=feat_vecs.shape[1]):
    """
    Iterate over values of k for chi2
    doing Xval for each k
    """
    #set_trace()
    r = []
    step_size = max_k/num_steps
    for k in range(1,max_k,step_size):
        r.append(xval(folds,get_best=False,k=k))
    print 'Features:'
    print [k for k in range(1,max_k,step_size)][1:]
    print 'F1:'
    print [f[0] for f in r][1:]
    print 'Accuracy'
    print [a[1] for a in r][1:]
    print 'Precision:'
    print [p[2] for p in r][1:]
    print 'Recall:'
    print [Re[3] for Re in r][1:]
# below are three different experiments
#shuffleTestSamples(10) # experiment increasing training data size
#searchK(10,25) #experiment to make a plot of metrics Vs num of features 
xval(10) # experiment to get the best results possible by selecting the model
         # with the best F1 score after iterating though k for each fold
