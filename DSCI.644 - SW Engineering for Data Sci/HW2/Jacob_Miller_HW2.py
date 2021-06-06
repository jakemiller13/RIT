#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 15:41:45 2020

@author: jmiller
"""

# =============================================================================
# Package Imports
# =============================================================================
import pandas as pd
import numpy as np
import re
from mpi4py import MPI
from math import floor
import argparse
import timeit

# =============================================================================
# Set MPI variables
# =============================================================================
comm = MPI.COMM_WORLD
status = MPI.Status()
size = comm.Get_size()
rank = comm.Get_rank()

# =============================================================================
# Command line inputs
# =============================================================================
try:
    ap = argparse.ArgumentParser()
    ap.add_argument("-s", "--seed",
                    nargs = '+',
                    required = True,
                    help = "Seed Files")
    ap.add_argument("-c", "--chr",
                    nargs = '+',
                    required = True,
                    help = "Chromosomes Files")
    args = vars(ap.parse_args())
except SystemExit:
    args = {'chr': ['/Users/jmiller/Google Drive/RIT/' + \
                    'DSCI.644 - SW Engineering for Data Sci/temp_chr1.fa'],
            'seed': ['/Users/jmiller/Google Drive/RIT/' + \
                     'DSCI.644 - SW Engineering for Data Sci/temp_faq1.fastq']}

# =============================================================================
# Check if master
# =============================================================================
if rank == 0:
    start_time = timeit.default_timer()
    results = []

# =============================================================================
# If worker is sending results, collect
# =============================================================================
    def receive_results():
        worker_counter = 0
        indices = []
        while True:
            msg = comm.recv(source = MPI.ANY_SOURCE,
                            status = status)
            print('Master received message from worker: {}'.
                  format(status.Get_source()))
            if status.Get_tag() == 1:
                indices.extend(msg)
            if status.Get_tag() == 0:
                print('Worker {} finished\n'.format(status.Get_source()))
                worker_counter += 1
                if worker_counter == size - 1:
                    print('Master completed, GOODNIGHT!')
                    end_time = timeit.default_timer()
                    run_time = end_time - start_time
                    indices.append('Workers: {} | Run time: {}'.
                                    format(size, run_time))
                    file_chr = args['chr'][0].split('/')[-1].split('.')[0]
                    file_seed = args['seed'][0].split('/')[-1].split('.')[0]
                    with open('Workers {} - {} - {}.txt'.
                              format(size, file_chr, file_seed), 'w') as file:
                        file.write('\n'.join(indices))
                    break

# =============================================================================
# Read in data (chromosome = reference, fastq = seed)
# =============================================================================
    regex = '.*N{10,}'

    # Chromosome files organized by columns
    df_chr = pd.DataFrame()
    
    for file in args['chr']:
        print('Reading in chromosome: {}'.format(file))
        df_chro = pd.read_csv(file)
        
        # Filter out lines that only contain >10 N (arbitrary cutoff)
        df_chro = df_chro[~df_chro[df_chro.columns[0]].str.match(regex)]
        # Capitalize all entries
        df_chro = df_chro[df_chro.columns[0]].str.upper()
        # Reset index for joining and subsequent slicing
        df_chro = df_chro.reset_index(drop = True)
        df_chr = df_chr.join(df_chro, how = 'outer')
    
    # Seeds files joined together
    df_faq = pd.DataFrame(columns = ['Read'])
    for file in args['seed']:
        print('Reading in seed: {}'.format(file))
        df_seed = pd.read_csv(file,
                              names = ['Read'],
                              skiprows = lambda x: (x - 1) % 4)
        df_faq = df_faq.append(df_seed)

# =============================================================================
# Prepare to split into even chunks
# =============================================================================
    read_count = floor(df_faq.shape[0] / size)

# =============================================================================
# Send 1 chromosome at a time
# =============================================================================
    for c in df_chr.columns:
        print('Sending chromosome: {}'.format(c))
        chromosome = df_chr[c].dropna()
        # Collapse chr into single bytearray for minimal memory usage
        chromosome = bytearray(''.join(chromosome.array), 'utf8')
        
# =============================================================================
# Divide into evenly sized chunks
# =============================================================================
        for worker in range(1, size):
            start = worker * read_count
            # Last chunk needs to account for any remaining reads
            if worker + 1 == size:
                stop = df_faq.shape[0] - 1
            else:
                stop = (worker + 1) * read_count
            df_read_chunk = df_faq.iloc[int(start): int(stop)]

# =============================================================================
# Send (chunk, chromosome) to workers
# ============================================================================= 
            print('Rank {} sending to worker: {}'.format(rank, worker))
            comm.send((df_read_chunk, chromosome),
                        dest = worker,
                        tag = 1)
        receive_results()
        
# =============================================================================
# Terminate all workers
# =============================================================================
        for worker in range(1, size):
            comm.send(None,
                      dest = worker,
                      tag = 666)

# =============================================================================
# If not master, then worker
# =============================================================================
else:
    while True:
        msg = comm.recv(source = 0,
                        status = status)
        try:
            df_chunk, chrom = msg
        except TypeError:
            pass
        
        # Terminate worker
        if status.Get_tag() == 666:
            print('Worker {} ending'.format(rank))
            break
            
        print('Worker {} received a msg from Master'.format(rank))
        
        # Find starting index of any read in chunk
        matches = df_chunk['Read'].apply(
            lambda x: [i.start() for i in
                       re.finditer(bytes(x, 'utf8'), chrom)])
        
        # Drop any empty cells
        matches = matches[matches.astype(bool)]
        
        # Collapse potential multiple counts to one single list of indices
        all_matches = [i for match in matches.to_list() for i in match]
        print('Matches: {}'.format(all_matches))
        
        # Send indices back to master, check for next task
        comm.send(all_matches,
                  dest = 0,
                  tag = 1)
        comm.send(None,
                  dest = 0,
                  tag = 0)
        print('Rank {} sent message to {}'.format(rank, status.Get_source()))