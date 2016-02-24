from sklearn import linear_model
import sklearn.datasets.svmlight_format as svmlight
from sklearn import grid_search
import cPickle

parameters = {'alpha':[.000001,.0000001,.00000001],'n_iter':[5000,10000,15000]}
feat_vecs,labels = svmlight.load_svmlight_file('featureFile.dat')
svr = linear_model.SGDClassifier()
svr.n_jobs = -1
clf = grid_search.GridSearchCV(svr, parameters)
clf.n_jobs = -1
clf.fit(feat_vecs,labels)

print clf.grid_scores_
with open('gridSearch.out', 'w') as fo:
    cPickle.dump(clf.grid_scores_,fo)
