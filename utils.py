import numpy as np

def create_warning_labels(y_true, window_size):
    """
    Generates warning labels (1) for 'window_size' days BEFORE a crash event.
    The actual crash event days are set to 0 to train the model to predict *before* the crash.
    
    Args:
        y_true (np.array): Binary array where 1 indicates a crash day.
        window_size (int): Number of days before the crash to label as 'warning'.
        
    Returns:
        np.array: Binary array with warning labels.
    """
    y_warning = np.zeros_like(y_true)
    true_event_indices = np.where(y_true == 1)[0]
    if not true_event_indices.any(): return y_warning
    
    event_groups = []
    current_event_start = true_event_indices[0]
    for i in range(1, len(true_event_indices)):
        if true_event_indices[i] > true_event_indices[i-1] + 1:
            event_groups.append(current_event_start)
            current_event_start = true_event_indices[i]
    event_groups.append(current_event_start) 
    
    for event_start_day in event_groups:
        warning_start = max(0, event_start_day - window_size)
        warning_end = event_start_day
        y_warning[warning_start : warning_end] = 1
        
    y_warning[y_true == 1] = 0
    return y_warning

def group_crash_events(y_true):
    """
    Identifies contiguous blocks of crash days (1s) in y_true.
    
    Returns:
        list of tuples: [(start_index, end_index), ...]
    """
    true_event_indices = np.where(y_true == 1)[0]
    if not true_event_indices.any(): return []
    
    # Find breaks in consecutive indices
    breaks = np.where(np.diff(true_event_indices) > 1)[0]
    starts = np.r_[true_event_indices[0], true_event_indices[breaks + 1]]
    ends = np.r_[true_event_indices[breaks], true_event_indices[-1]]
    
    return list(zip(starts, ends))