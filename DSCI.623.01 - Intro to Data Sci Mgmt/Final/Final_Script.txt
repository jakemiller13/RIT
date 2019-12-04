#!/usr/local/bin/bash

#########################################################################
# QUESTION 1: How many records are there in the dataset? 		#
#########################################################################

# Since we converted a CSV file to JSON, we know "file" is carried
# through every record. Therefore we can just do an egrep to locate
# all instances of "type (note the quotation mark is intentional
# to make sure I specifically capture the key "type" and not the
# word (type) somewhere else). If we are concerned the exact instance
# "type shows up somewhere else, we could cross reference with any
# other key.
# Alternatively, if we are just working with the CSV file (as I will be
# from here on out), we can easily use the number-lines command, but
# this prints out each and every line, so I've commented it out here
# to keep the actual standard output to a minimum if the script is run.

########
# CODE #
########

# number-lines -H propublica_trump_spending-1.csv

printf "\n"

echo "--- Q1: Number of records ---"
egrep \"type < propublica_json.json | wc -l

printf "\n"

#@@@@@@@@#
# ANSWER #
#--------#
#  1193  #
#@@@@@@@@#

#########################################################################
# QUESTION 2: How many actual unique "purposes" are there for this 	#
# spending? And list what they are					#
#########################################################################

# The following pipeline first gives the count of unique purposes,
# and then lists them all out. The "sed" command is there to remove
# the header line so I don't accidentally count it as a unique entry.

########
# CODE #
########

echo "--- Q2: Number of unique purposes ---"
csv2tsv propublica_trump_spending-1.csv | tsv-select -f 5 | sed "1d" | tsv-uniq | wc -l

printf "\n"

echo "--- Q2: List of unique purposes ---"
csv2tsv propublica_trump_spending-1.csv | tsv-select -f 5 | sed "1d" | tsv-uniq

printf "\n"

#@@@@@@@@@@@#
#  ANSWER   #
#-----------#
# unique: 8 #
#-----------#
# Lodging   #
# Rent      #
# Event     #
# Travel    #
# Payroll   #
# Food      #
# Other     #
# Legal     #
#@@@@@@@@@@@#

#########################################################################
# QUESTION 3: How much is is being spent on each unique			#
# "purpose_scrubbed"? 							#
#########################################################################

# This uses the "purpose_scrubbed" column and sums all of values in the
# "amount" column.

########
# CODE #
########

echo "--- Q3: Spending per purpose ---"
csv2tsv propublica_trump_spending-1.csv | tsv-summarize -H -g 5 --sum 4 | tsv-pretty -f -u

printf "\n"

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#            ANSWER            #
#------------------------------#
# purpose_scrubbed  amount_sum #
# ----------------  ---------- #
# Lodging            881461.37 #
# Rent              2956588.00 #
# Event             2132210.96 #
# Travel            9282248.45 #
# Payroll            424615.83 #
# Food               145054.51 #
# Other               95716.56 #
# Legal              168016.00 #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#########################################################################
# QUESTION 4: How many unique "property_scrubbed" entries are there?	#
# And how many of these actually contain the word "Trump"?		#
#########################################################################

# The first line below shows how many unique entries there are in
# "property_scrubbed".
# The second line shows how many unique entries specifically contain
# "Trump" (case-insensitive).
 
########
# CODE #
########

echo "--- Q4: Number of unique properties ---"
csv2tsv propublica_trump_spending-1.csv | tsv-select -f 6 | sed "1d" | tsv-uniq | wc -l

printf "\n"

echo "--- Q4: Unique properties that contain \"Trump\" ---"
csv2tsv propublica_trump_spending-1.csv | tsv-select -f 6 | sed "1d" | tsv-uniq | grep -i Trump | wc -l

printf "\n"

#@@@@@@@@@@@@#
#   ANSWER   #
#------------#
# unique: 38 #
# trump: 33  #
#@@@@@@@@@@@@#

#########################################################################
# QUESTION 5: What is the list of how much is being spent at each 	#
# unique "property_scrubbed" entry?					#
#########################################################################

# This selects just the "amount" and "property_scrubbed" columns,
# groups them by "property_scrubbed" and then sums the amounts.
# Note that this includes the ENTIRE list, not just properties that
# contain "Trump"

########
# CODE #
########

echo "--- Q5: Spending per property ---"
csv2tsv propublica_trump_spending-1.csv | tsv-select -H -f 6,4 | tsv-summarize -H --group-by 1 --sum 2 | tsv-pretty -f -u

printf "\n"

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#
#                     ANSWER                     #
#------------------------------------------------#
# property_scrubbed                   amount_sum #
# -----------------                   ---------- #
# Trump Hotel NY                        79663.26 #
# Trump Tower Commercial LLC          2700562.98 #
# The Trump Corporation                250621.34 #
# Trump Soho NY                         23000.34 #
# Tag Air, Inc.                       9252770.40 #
# Trump CPS LLC                         72000.00 #
# Trump Plaza LLC                      181250.00 #
# Trump Payroll Corp                   218546.71 #
# Trump Cafe NY                          2946.73 #
# Trump Restaurants LLC                262097.32 #
# Trump Hotel Las Vegas                535420.07 #
# Trump Grill NY                         1121.56 #
# Trump Hotel D.C.                     141462.99 #
# Trump Golf Club Miami                444418.73 #
# Trump Hotel Chicago                   55664.17 #
# Trump Ice LLC                          5165.41 #
# Eric Trump Wine Manufacturing, LLC    38552.10 #
# Trump Virginia Acquisitions, LLC       3669.68 #
# Trump Golf Club Jupiter               61041.80 #
# Trump Golf Club Palm Beach           163055.66 #
# Trump Golf Club Bedminster            77066.94 #
# Trump Golf Club Westchester           48239.77 #
# Other                                 58464.34 #
# Trump Golf Club Charlotte             18279.67 #
# Trump Golf Club D.C                   42219.28 #
# Trump Hotel D.C                      848886.34 #
# BLT Prime D.C                         61463.83 #
# Trump Golf Club D.C.                    214.42 #
# Trump Golf Club Bedminster            15221.10 #
# Benjamin Bar & Lounge D.C              1001.40 #
# Trump Golf Club L.A.                   8045.58 #
# Mar-a-Lago Club LLC                  295125.02 #
# Trump Hotel Honolulu                   1597.13 #
# Trump Golf Resort Scotland            13307.85 #
# Trump Hotel Panama                    20778.67 #
# Trump Golf Club                       16625.80 #
# Trump Hotel Vancouver                 38618.97 #
# Trump Golf Club Doonberg              27724.32 #
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@#

#########################################################################
# QUESTION 6: What is the total being spent on these properties?	#
#########################################################################

# This simply sums the entire "amount" column

########
# CODE #
########

echo "--- Q6: Total spent at Trump properties ---"
csv2tsv propublica_trump_spending-1.csv | tsv-summarize -H --sum 4

printf "\n"

#@@@@@@@@@@@@@#
#   ANSWER    #
#-------------#
# amount_sum  #
# 16085911.68 #
#@@@@@@@@@@@@@#
