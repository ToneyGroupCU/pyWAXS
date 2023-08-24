# source: http://stackoverflow.com/a/1557906/6009280
# used to get clock time for program execution
import atexit
import time
from time import process_time # note: time.clock deprecated for Python 3.7 and below
from time import process_time_ns
from functools import reduce
 
def seconds_to_str(t):
    return "%d:%02d:%02d.%03d" % \
           reduce(lambda ll, b: divmod(ll[0], b) + ll[1:],
                  [(t * 1000,), 1000, 60, 60])
 
line = "=" * 40
 
def log(s, elapsed=None):
    print(line)
    print(seconds_to_str(process_time()), '-', s)
    if elapsed:
        print("Elapsed time:", elapsed)
    print(line)
    print()
 
def endlog():
    end = process_time()
    elapsed = end - start
    log("End Program", seconds_to_str(elapsed))
 
def now():
    return seconds_to_str(process_time())

def now_ns():
    return seconds_to_str(process_time_ns())
 
start = process_time()
atexit.register(endlog)
log("Start Program")