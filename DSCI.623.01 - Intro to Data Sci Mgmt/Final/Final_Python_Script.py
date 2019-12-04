#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 14:48:01 2019

@author: Jake
"""

import pandas as pd
import json

csvFilePath = 'propublica_trump_spending-1.csv'
jsonFilePath = 'propublica_json.json'

csv_file = pd.read_csv(csvFilePath)
json_file = json.loads(csv_file.to_json(orient = 'records'))

with open(jsonFilePath, 'w') as file:
    file.write(json.dumps(json_file, indent = 4))