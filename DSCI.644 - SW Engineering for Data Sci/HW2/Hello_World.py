#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 16:45:50 2020

@author: jmiller
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print('Size: {}'.format(size))

if rank != 0:
    print('Hello! My rank is: {}'.format(rank))
else:
    print('Hello! My rank is: {}'.format(rank))