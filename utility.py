"""
Utility functions for logging and profiling
"""

import time
import logging

def profile(func):
    """
    profile decorator to log the time taken by a function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logging.info("Function %s took %s seconds", func.__name__, end - start)
        return result
    return wrapper