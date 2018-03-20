import csv
import collections

from data.raw_data import *

def line_count(filename):
    result = 0
    for i, row in enumerate(csv.DictReader(open(filename)), start=1):
       result += 1
    return result

def a99f214a_count(filename):
    result = 0
    for i, row in enumerate(csv.DictReader(open(filename)), start=1):
        if row['device_id'] == 'a99f214a':
            result += 1
    return result

def is_hour_increase(filename):
    prev_hour = 0
    for i, row in enumerate(csv.DictReader(open(filename)), start=1):
        if int(row['hour']) < prev_hour:
            return False
        prev_hour = int(row['hour'])
    return True

def device_id_count(filename):
    value_count = collections.defaultdict(int)
    for row in csv.DictReader(open(filename)):
        value_count[row['device_id']] += 1
    return value_count

if __name__ == '__main__':
    pass