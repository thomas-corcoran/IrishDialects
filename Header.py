from collections import OrderedDict
class Header(OrderedDict):
    
    def __init__(self):
        OrderedDict.__init__(self)
        self.d = {}
        self.count =1 
    def add(self,feature):
        feature = unicode(feature)
        if feature not in self:
            self[feature] = self.count
            self.count+=1

