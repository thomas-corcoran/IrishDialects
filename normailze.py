from __future__ import print_function
import sys

def warning(*objs):
        print("evaluate.py: WARNING: ", *objs, file=sys.stderr)
def normalize(dat_file,out_file):    
    """Normalize svm-light formated file between [0,1]
    sklearn.preprocessing has issues with svm-light formatted arrays
    """
    maxVal = 0
    with open(dat_file) as fi:
        # find our largest number
        print("Reading from: {}".format(dat_file))
        for line in fi:
            l = line.split()
            for i in l[1:]:
                num = float(i.split(':')[1])
                if num > maxVal:
                    maxVal = num
    with open(dat_file) as fi:
        # normalize our data by dividing by maxVal
        print("Fitting data...")
        lines = []
        for line in fi:
            l = line.split()
            for i,val in enumerate(l[1:], start = 1):
                l[i] = val.split(':')[0] + ':' + str(float(val.split(':')[1])/maxVal)
            lines.append(' '.join(l))
    with open(out_file, 'w') as fo:
        print("Writing to: {}".format(out_file))
        fo.write('\n'.join(lines))
if __name__ == '__main__':
    if len(sys.argv)<3:
        warning("Need input and output file names!")
        sys.exit(1)
    else:
        dat_file = sys.argv[1]
        out_file = sys.argv[2]
        normalize(dat_file,out_file)