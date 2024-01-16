# -*- coding: utf-8 -*-

from utils.sur_analysis import *
from utils.myutils import provide_determinism
import argparse

if __name__ == '__main__':
    provide_determinism(0)
    parser = argparse.ArgumentParser(description='SubtypeWGME v1.0')
    parser.add_argument("-t", dest='type', default="all", help="cancer type,default: all")
    parser.add_argument("-n", dest='n_cluster', type=int, default=3, help="cancer cluster number")
    parser.add_argument("-e", dest='exam', default='get_p', help="")
    parser.add_argument("-m", dest='method', default='SubtypeWGME', help="")
    parser.add_argument("-metric", dest='metric', default='euclidean', help="")
    args = parser.parse_args()
    print(args)
    if args.exam == 'assign_p':
        get_p_assign(cancer_type=args.type, method=args.method)  # get assigned p for all cancer or one cancer
    elif args.exam == 'get_p':
        print(get_p(cancer_type=args.type, method=args.method))  # only get assigned p for one cancer
    elif args.exam == 'cc':
        cc(cancer_type=args.type, method=args.metric)            # assigned and auto suggest
    elif args.exam == 'proba':
        get_proba()                                              # get probability
    else:
        print('unsupported')
