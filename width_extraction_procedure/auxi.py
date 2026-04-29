import os
import re
from datetime import datetime,timedelta
def Filename2Numbers(path):
    """
    Extract the first two numeric fields from a file path.
    Returns (num1, num2) as integers.
    Returns None if fewer than two numeric fields are found.
    """
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    
    parts = filename.split('_')
    
    numeric_parts = [p for p in parts if p.isdigit()]
    
    if len(numeric_parts) >= 2:
        return int(numeric_parts[0]), int(numeric_parts[1])
    else:
        return None
def Filename2Datetime(path):
    
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    
    time_pattern = r'(\d{8}T\d{6})'
    time_matches = re.findall(time_pattern, filename)
    
    if len(time_matches) != 2:
        raise ValueError(f"Expected 2 time strings in filename, found {len(time_matches)}")

    start_datetime = datetime.strptime(time_matches[0], '%Y%m%dT%H%M%S')
    end_datetime = datetime.strptime(time_matches[1], '%Y%m%dT%H%M%S')
    time_diff = end_datetime - start_datetime
    return start_datetime + time_diff / 2 # Mean datetime
def Dist2Q(dist):
    """
    Abandoned
    """
    if dist<10000:
        return 0
    if dist<30000:
        return 1.05-1050/(dist-9000)
    if dist<50000:
        return 1
    return 1.05-1050/(71000-dist)
import numpy as np
