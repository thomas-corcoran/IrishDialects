from collections import Counter
from CorpusReader import CorpusReader
from pdb import set_trace
import json
from glob import iglob
'''
M = CorpusReader('munster')
C = CorpusReader('connacht')
U = CorpusReader('ulster')
print "Done."
print "Creating Balanced Set"

l = [U,M,C]
l.sort(key=lambda x: x.countSentences)
for f in l:
    f.truncateSentenceList(l[-1].countSentences())
'''
suff = ('eas', 'is', 'ais', 'as', 'eamar', 'amar', 'eabhair', 'abhair', 'eadar', 'adar')


def isVerb(POS):
    for pos in POS.split('|'):
        if pos.startswith('V'):
            return True
    return False
def checkSuff(sentence):
    suff_cnt = 0
    for word in sentence:
        if isVerb(word['POS']):
            if word['token'].endswith(suff):
                suff_cnt += 1
    return suff_cnt
def checkAfter(sentence):
    after_cnt = 0
    for word in sentence:
    #set_trace()
        if word['lemma'].lower() == "diaidh-n":
            after_cnt += 1
    return after_cnt
def checkLenitedN(sentence):
    lenited = 0
    eclip = 0
    leng = len(sentence)
    for i,word in enumerate(sentence):
        if word["POS"].startswith('S') and i+2 <leng:
            if sentence[i+1]['POS'].startswith("T"):
                if sentence[i+2]['POS'].startswith('N'):
                    if sentence[i+2]['token'].lower() != sentence[i+2]['lemma'].split('-')[0].lower():
     #                   set_trace()
                        if sentence[i+2]['token'].lower()[0] !=sentence[i+2]['lemma'].split('-')[0].lower()[0]:
                            eclip+=1
                        elif sentence[i+2]['token'].lower()[1] == 'h':
                            lenited +=1
    return eclip
    return lenited

def ensureDate(j):
    if j['pubdate'].startswith('19'):
        return True
    if j['time'].startswith('19'):
        return True
    if j['time'].startswith('20'):
        return True
    if j['pubdate'].startswith('20'):
        return True
    return False

def run():
    d = {}

    books = iglob('json_files/*-meta.json')
    i=0
    for b in books:
        print str(i)
        i+=1
        with open(b) as fi:
            bk = json.load(fi)
        if bk['birthplace'] and ensureDate(bk):
            suff_cnt = 0
            after_cnt = 0
            len_cnt = 0
            with open(b.split('-')[0]+'.json') as fi:
                book = json.load(fi)
            for sentence in book['sentences']:
                suff_cnt += checkSuff(sentence)
                after_cnt += checkAfter(sentence)
                len_cnt += checkLenitedN(sentence)
            if book['birthplace'] not in d:
                d[book['birthplace']] = {}
                d[book['birthplace']]['wordcounts'] = int(book['wordcount'])
                d[book['birthplace']]['suff_count'] = suff_cnt
                d[book['birthplace']]['after_count'] = after_cnt
                d[book['birthplace']]['lenited_count'] = len_cnt

            else:
                d[book['birthplace']]['wordcounts'] += int(book['wordcount'])
                d[book['birthplace']]['suff_count'] += suff_cnt
                d[book['birthplace']]['after_count'] += after_cnt
                d[book['birthplace']]['lenited_count'] = len_cnt


               
    with open('suffCount.json', 'wb') as fo:

        json.dump(d,fo)
if __name__== "__main__":
    run()
