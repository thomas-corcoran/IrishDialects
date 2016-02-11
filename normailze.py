import sys
from pdb import set_trace
dat_file = sys.argv[1]
out_file = sys.argv[2]
#set_trace()
maxVal = 0
with open(dat_file) as fi:
    for line in fi:
        l = line.split()
        for i in l[1:]:
            num = float(i.split(':')[1])
            if num > maxVal:
                maxVal = num
#set_trace()
with open(dat_file) as fi:
    lines = []
    for line in fi:
        l = line.split()
        for i,val in enumerate(l[1:], start = 1):
            l[i] = val.split(':')[0] + ':' + str(float(val.split(':')[1])/maxVal)
        lines.append(' '.join(l))
#set_trace()
with open(out_file, 'w') as fo:
    fo.write('\n'.join(lines))
