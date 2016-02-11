from sklearn.cross_validation import StratifiedKFold, KFold, LeaveOneOut
from multiprocessing import Pool

class XValMT(StratifiedKFold, KFold, LeaveOneOut):
    
    def run(self,fn):
        pool = Pool(processes=self.n_folds)
        self.results = pool.map(fn, [(trainI,testI) for trainI,testI in self])



