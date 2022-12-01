# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 15:57:59 2022

@author: lvyang
"""

for k in ["positive","negative"]:
    f=open("%s.txt"%k,"r")
    f1=open("%s.fasta"%k,"w")
    for line in f:
        f1.write(">1\n"+line)
    f.close()
    f1.close()

    