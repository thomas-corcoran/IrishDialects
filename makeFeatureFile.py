from itertools import chain
from CorpusReader import CorpusReader
from levenshtein import levenshtein as lev
from collections import Counter 
from pdb import set_trace
from Header import Header

header = Header()
SUFF = ('eas', 'is', 'ais', 'as', 'eamar', 'amar', 'eabhair', 'abhair', 'ead    ar', 'adar')

books = True
sentences = False

print "Loading Corpora..."
if books:
    print "\tloading munster"
    M = CorpusReader('munster')
    print "\tloading connacht"
    C = CorpusReader('connacht',M.countBooks())
    C.truncateBooks(M.countBooks())
    print "\tloading ulster"
    U = CorpusReader('ulster', M.countBooks())
    U.truncateBooks(M.countBooks())
#print "Done."
if sentences:
    print "Creating Balanced Set of sentences"
    l.sort(key=lambda x: x.countBooks())
    M = CorpusReader('munster')
    C = CorpusReader('connacht')
    U = CorpusReader('ulster')
    MIN_LENG = min([x.countSentences() for x in l])
    for x in l:
        x.truncateSentences(MIN_LENG)
print "Done!"
l = [U,M,C]
def makeRow(cnt,header,dialect):
    l = []
    if dialect == 'M': dialect = 1
    if dialect == 'C': dialect = 2
    if dialect == 'U': dialect = 3
    c = sorted(cnt.items(),key = lambda x: int(x[0]))
    for k in c:
        l.append("%s:%f"%(k[0],k[1]))
    return str(dialect)+" "+' '.join(l)+'\n'

def isVerb(POS):
    for pos in POS.split('|'):
        if pos.startswith('V'):
            return True
    return False
def checkSuff(sentence):
    suff_cnt = 0 
    for word in sentence:
        if isVerb(word['POS']):
            if word['token'].endswith(SUFF):
                suff_cnt += 1
    return suff_cnt
def checkLenitedN(sentence):
    lenited = 0
    eclip = 0
    leng = len(sentence)
    for i,word in enumerate(sentence):
        if word["POS"].startswith('S') and i+2 <leng:
            if sentence[i+1]['POS'].startswith("T"):
                if sentence[i+2]['POS'].startswith('N'):
                    if sentence[i+2]['token'].lower() != \
                            sentence[i+2]['lemma'].split('-')[0].lower():
     #                   set_trace()
                        if sentence[i+2]['token'].lower()[0] != sentence[i+2]['lemma'].split('-')[0].lower()[0]:
                            eclip+=1
                        elif sentence[i+2]['token'].lower()[1] == 'h':
                            lenited +=1
    return (eclip,lenited)

featureFile = []
print "Creating Feature File"
header.add("EMPTY")
header.add("ECLIPSIS")
header.add("LENITION")
header.add("SUFFIX_COUNT")
with open('featureFile.dat', 'w') as fo:
    i = 0
    for dialect in l:
        for book in dialect:
            cnt = Counter()
            for sentence in book['sentences']:
                suff = checkSuff(sentence)
                cnt[header["SUFFIX_COUNT"]] = suff
                e_l = checkLenitedN(sentence)
                cnt[header["ECLIPSIS"]] = e_l[0]
                cnt[header["LENITION"]] = e_l[1]
                for word in sentence:
                    #if isVerb(word['POS']):
                    header.add(word['token'])
                    cnt[header[word['token']]] += 1

            if len(cnt) == 0:
                cnt[header['EMPTY']] = 0
            i+=1
            print str(i)
            try:
                s = makeRow(cnt,header,book['dialect'][0])
            except:
                set_trace()
            fo.write(s)
print "Done!"
