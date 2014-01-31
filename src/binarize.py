'''
#==============================================================================
hw2.py
/Users/aelshen/Documents/Dropbox/School/CLMS 2013-2014/Winter 2014/Ling 571-Deep Processing Techniques for NLP/src/hw2.py
Created on Jan 16, 2014
@author: aelshen
#==============================================================================
'''

import os
import sys

#==============================================================================
#--------------------------------Constants-------------------------------------
#==============================================================================
DEBUG = True
#==============================================================================
#-----------------------------------Main---------------------------------------
#==============================================================================
def main():
    if(len(sys.argv) < 2):
        print "Requires 2 arguments:(1)original vector file (2)output file"
        sys.exit()
    original_file = open(sys.argv[1],'r')
    binary_file = open(sys.argv[2],'w')
    
    for line in original_file.readlines():
        features = line.split()

        binary = []
        binary.append(features[0])
        i = 1
        while (i<len(features)):
            x = features[i].split(':')
            x[1] = "1"
            binary.append( ':'.join(x) )
            i+=1
        binary_file.write( " ".join(binary) )
        binary_file.write(os.linesep)
    
    original_file.close()
    binary_file.close()
    print "Hello, World!"

#==============================================================================    
if __name__ == "__main__":
    sys.exit( main() )