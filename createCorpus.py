from __future__ import print_function
import sys
import os
from re import compile
import json
def warning(*objs):
    print("WARNING: ", *objs, file=sys.stderr)
    
def makeJSONS(vert_file):
    corpus_files_dir = 'corpus_files/'
    if not os.path.exists(corpus_files_dir ):
        os.makedirs(corpus_files_dir)
    i = 0
    with open(vert_file) as fi:
        line = fi.readline()
        for _ in xrange(39008721):
            if line.startswith('<doc id='):
                print( i)
                i+=1
                meta = {'doc id':"", 'title':"", 'author':"", 
                    'nativespeaker':"", 
                    'dialect':"", 'medium':"", 'genre':"", 'genre2':"", 
                    'pubdate':"", 'translation':"", 'origtitle':"", 
                    'origauthor':"", 'time':"", 'authordob':"", 
                    'birthplace':"", 'residence':"", 'wordcount':""}
                for dp in meta:
                    pat = compile(r'%s="(.*?)"'%dp)
                    meta[dp] = pat.findall(line)[0]
                sentences = [] # all sentences in doc go here
                line = fi.next()
                #set_trace
                while not line.startswith('<doc id='):
                    sentence = []   # words for a sentence go here 
                    while not line.startswith('</s>'):
                        if line.startswith('<doc id='):
                            break
                        elif line.startswith('<s>'):
                            line = fi.next()
                            break
                        else:
                            word={'token':'','POS':'','lemma':''}
                            l = line.split()
                            #set_trace()
                            if len(l) == 3:
                                word['token'],word['POS'],word['lemma'] = l
                                sentence.append(word)
                                line = fi.next()
                            else:
                                line = fi.next()
                           
                    #set_trace()
                    if not line.startswith('<doc id='):
                        try: line = fi.next()
                        except  StopIteration: break 
                    if sentence:
                        sentences.append(sentence)
#                    set_trace()
                d = meta
                with open(corpus_files_dir + d['doc id']+'-meta.json', 'w') as fo:
                    json.dump(meta,fo)
                d['sentences'] = sentences
                with open(corpus_files_dir + d['doc id']+'.json', 'w') as fo:
                    json.dump(d,fo)
if __name__ == '__main__':
    try:
        makeJSONS(sys.argv[1])
    except IndexError:
        warning("Must be called with location to vert file")
        sys.exit(1)