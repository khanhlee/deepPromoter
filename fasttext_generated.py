# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 15:22:57 2018

@author: khanhle
"""
import re
import sys

in_file = sys.argv[1]
out_file = sys.argv[2]

fout = open(out_file,'w')

spe_character = []

with open(in_file) as f:
    for line in f:
        if (line.startswith('>') == False) and (any(x in line for x in spe_character) == False):
            sequence = ''.join(line).replace('\n','')
            print(sequence)
            fout.write(' '.join(list(sequence)) + '\n')
                   
fout.close()
