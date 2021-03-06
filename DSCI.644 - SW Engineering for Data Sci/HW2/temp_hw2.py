import numpy as np
import argparse
import os
import re
import time
from mpi4py import MPI

mpi_comm  = MPI.COMM_WORLD
status    = MPI.Status()
comm_size = mpi_comm.Get_size()
rank      = mpi_comm.Get_rank()

regex = re.compile('[+@_!#$%^&*()<>?/\|}{~:]')


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--seed", nargs='+',  required=True, help="Seed Files")
ap.add_argument("-c", "--chr" , nargs='+',  required=True, help="Chromosomes Files")
args = vars(ap.parse_args())

## Master Process ##
if rank==0:
    all_results = {}

    # A function to revieve results as they arrive from workers
    def receive_results(chr_name):
        result = {}
        worker_count = 0
        while True:   # this function in th Master will loop till workers finish sending all their results
            ###################################################
            ###### Put a Recv() returns to the msg list ######
            ###################################################
            msg = ### Put Recieve Function ###               # receiving results from ANY Worker
            print "Master received a msg from Worker {} with tag: {}".format(status.Get_source(), status.Get_tag())
            if status.Get_tag()==0:                         # Worker is saying the task is done
                worker_count+=1
                print "Master: Worker {} finshed a task".format(status.Get_source())
                if worker_count==comm_size-1: break         #loop breaks when all Workers say they are done
            elif status.Get_tag()==1:                       # Worker is saying 'here are some results'
                if result.has_key(msg[0]):                  # Registering ther results from a Worker
                    result[msg[0]]+=msg[1]
                else:
                    result[msg[0]] = msg[1]
        all_results[chr_name] = result


    for seed_file in args['seed']:                  # Interating through the Seed files
        with open(seed_file, 'rb') as seedF:        # Reading a Seed file
            seed = [x[:-1].upper() for x in seedF if regex.search(x) == None]

        print "Master: Seed {} started".format(seed_file)

        for chr_file in args['chr']:                # Interating through the Chromosomes
            with open(chr_file, 'rb') as chrF:      # Reading a Chromosome file
                chr_name = chrF.readline()[1:]                              # Reading the Chromosome name from the file's header
                chr = chrF.read().replace('\n','').upper().replace('N', '') # turing the Chromosome to one string & removing the mysterious N protien
            ###################################################             # spliting the Seed file two equal chuncks to send them to the worker processes
            ######  Split the seed file to equal chuncks ######
            ######  (to send each chunck to a worker)    ######
            ###################################################
            for i in xrange(comm_size-1):
                #####################################################################
                ###### Put a Send() to send the chr and the seed-chuck as pair ######
                #####################################################################
                ## Put a Send Function to the Workers                       # sending data to worker processes
            receive_results(chr_name)                                       # recieving results
            print "Master:---> completed Chromosome {} completed".format(chr_file)
        print "Master: Seed {} finished".format(seed_file)
        print "------------------------"

    #preparing for the end of the program: Telling Workers to go to sleep
    for r in range(1,comm_size,1):
        ###################################################################
        ###### Put a Send() to send termination messages to Wrokers ######
        ###################################################################
    print "Master finished work and is leaving... Bye!"

    ###########################################
    ###### Write Final Results in a file ######
    ###########################################



## Worker Processes ##
else:
    while True:                                 # Worker will keep spining till Master say there is not more work (files) to process
        msg = mpi_comm.recv(source=0, status=status)
        seed, chr = msg
        if status.Get_tag() == 0:       # 'Go to sleep worker'
            print "Worker {} received a tremination tag from Master... Bye!".format(rank)
            break   # Worker finished its work when Master say that the work is done
        print "Worker {} received a msg from Master!".format(rank)
        genomes = {}
        for g in seed:                          # fill the hash-table of the genomes from the Seed (read) chuncks
            if genomes.has_key(g):
                genomes[g]+=1
            else:
                genomes[g]=1

        ###############################################################
        ###### You can change the following part if you want to  ######
        ###### consider the N's in the Chromosome and Seed files ######
        ###############################################################
        for g, count in genomes.iteritems():    # Search for the indecies of the genomes in the Chromosome
            indcies = [i.start() for i in re.finditer(g, chr)]
            for i in indcies:                   # Send the indices to the Master
                mpi_comm.send([i, count], dest=0, tag=1)

        print "Worker {} finished its task and ready for the next!".format(rank)
        mpi_comm.send(None, dest=0, tag=0)      #  Worker finished task and informing Master about it
