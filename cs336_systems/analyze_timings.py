import pickle
import sys

def analyze_timing(filepath):
    print(f"Analyzing timings for {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    if "device_traces" not in data:
        return

    # Extract all events with timestamps
    events = []
    raw_traces = data["device_traces"]
    for item in raw_traces:
        if isinstance(item, list):
            events.extend(item)
        else:
            events.append(item)
    
    # Filter valid events
    valid_events = []
    for e in events:
        if isinstance(e, dict) and "action" in e and "time_us" in e:
            valid_events.append(e)
    
    # Sort by time
    valid_events.sort(key=lambda x: x["time_us"])
    
    if not valid_events:
        print("No valid events found.")
        return

    # Try to identify cycles based on memory usage
    current_mem = 0
    peak_mem = 0
    start_time = valid_events[0]["time_us"]
    
    # We want to find the time spread of the "Up" phase vs "Down" phase.
    # Heuristic: 
    # Up Phase: Net positive allocations until peak.
    # Down Phase: Net negative allocations until baseline.
    
    # Let's track memory over time buckets to visualize slope
    min_time = valid_events[0]["time_us"]
    max_time = valid_events[-1]["time_us"]
    duration = max_time - min_time
    print(f"Total Duration: {duration / 1e6:.4f} seconds")
    
    # Let's count cycles by looking for "Bottoms" (local minima)
    # A generic approach: 
    # 1. Calculate memory curve.
    # 2. Find peaks and valleys.
    # 3. Measure time Peak->Valley vs Valley->Peak.
    
    memory_curve = []
    times = []
    
    first_ts = valid_events[0]["time_us"]
    print(f"First timestamp: {first_ts}")
    # Inspect if there are initial snapshot segments
    
    current_mem = 0
    
    # Check if we should offset by min_mem logic or if there are "OOM" segments
    # Actually, let's just use the relative peak (Peak - Baseline) + Theoretical Baseline
    
    for e in valid_events:
        size = e.get("size", 0)
        action = e.get("action", "")
        
        if action == "alloc":
            current_mem += size
        elif action == "free_completed":
            current_mem -= size
        
        memory_curve.append(current_mem)
        times.append(e["time_us"])
        
    print(f"Memory curve length: {len(memory_curve)}")
    print(f"Sample mems: {memory_curve[::1000]}")
    print(f"Min mem: {min(memory_curve)}, Max mem: {max(memory_curve)}")
    baseline = min(memory_curve)
    global_peak = max(memory_curve)
    print(f"Baseline: {baseline}, Peak: {global_peak}")

    # Find approximate steps (assuming 5 steps)
    # Peak is when mem is high. Valley is when mem is low.
    # Let's segment by "return to baseline".
    # Baseline is roughly the minimum seen (or close to it).
    
    baseline = min(memory_curve)
    
    # Identify "cycles" where memory goes high then returns to near baseline
    # We can detect state changes: "Rising", "Falling"
    
    cycles = []
    cycle_start_t = times[0]
    peak_t = times[0]
    peak_val = 0
    
    # Simple state machine
    # Wait for rise > threshold
    # Wait for fall < threshold
    
    # Find max peak to set threshold
    global_peak = max(memory_curve)
    threshold = (global_peak - baseline) * 0.1 + baseline # 10% above baseline
    
    in_cycle = False
    
    peak_times = []
    valley_times = []
    
    # Only capturing explicit peaks/valleys might be noisy.
    # Let's just print the time difference between major events.
    
    # Alternative: Print the timestamp of every 1000th event to see density?
    # No, let's just find the first full cycle.
    
    # Manual scan for the first major peak
    curr_max = 0
    curr_max_t = 0
    
    # We expect 5 peaks.
    # Let's find 5 highest points nicely separated.
    # ...
    
    # Actually, simpler check:
    # Calculate "Time spent executing allocs" vs "Time spent executing frees"?
    # No, they are interleaved.
    
    # Let's just output the curve as (time_offset, mem_mb) to CSV for a quick check or print summary.
    # Or just average rise time vs average fall time.
    
    rises = []
    falls = []
    
    # Detect segments
    # Segment = monotonically increasing (roughly) or decreasing?
    # The curve is sawtooth.
    
    # Let's find the indices of local maxima and minima.
    peaks = []
    valleys = []
    
    # Smoothing? No need.
    # Just iterate.
    
    # Find points where we are "Low" (< baseline + 1GB) and "High" (> 80% peak)
    high_mark = baseline + 0.8 * (global_peak - baseline)
    low_mark = baseline + 0.2 * (global_peak - baseline)
    
    state = "low" # or "rising", "high", "falling"
    last_low_t = times[0]
    last_high_t = None
    
    # Detect simple min/max in windows
    # Since we know there are 5 steps, let's just find the global timeline and visually segment it in logs.
    
    # Print start/end of "alloc" streaks vs "free" streaks?
    # Better: Detect change of sign in memory derivative.
    
    times_us = [e["time_us"] for e in valid_events]
    mems = memory_curve
    
    # Find peaks (local maxima)
    # A peak is a point greater than neighbors.
    
    peaks = []
    valleys = []
    
    # Smooth a bit or just look for substantial extrema
    for i in range(1, len(mems)-1):
        if mems[i] > mems[i-1] and mems[i] > mems[i+1]:
            # Potential peak
            if mems[i] > high_mark:
                peaks.append((times_us[i], mems[i]))
        elif mems[i] < mems[i-1] and mems[i] < mems[i+1]:
             if mems[i] < low_mark:
                 valleys.append((times_us[i], mems[i]))
                 
    # Filter close peaks (keep highest in window)
    # For now, just print the raw candidates
    print(f"Candidate Peaks: {len(peaks)}")
    print(f"Candidate Valleys: {len(valleys)}")
    
    # Find peaks using a window approach to ignore noise
    # We want 5 peaks.
    window_size = 500 # events
    peaks = []
    
    for i in range(len(mems)):
        start = max(0, i - window_size)
        end = min(len(mems), i + window_size)
        window = mems[start:end]
        if mems[i] == max(window) and mems[i] > baseline + (global_peak - baseline)*0.5:
             peaks.append((times_us[i], mems[i]))
             
    # Deduplicate peaks that are close in time
    unique_peaks = []
    if peaks:
        peaks.sort(key=lambda x: x[0])
        current_peak = peaks[0]
        for p in peaks[1:]:
            if p[0] - current_peak[0] < duration * 0.1: # if within 10% of total duration
                if p[1] > current_peak[1]:
                    current_peak = p
            else:
                unique_peaks.append(current_peak)
                current_peak = p
        unique_peaks.append(current_peak)
        
    real_peaks = sorted(unique_peaks, key=lambda x: x[1], reverse=True)[:5]
    real_peaks.sort(key=lambda x: x[0])
    print(f"Top 5 Peaks (Time, Mem): {real_peaks}")
    
    rises = []
    falls = []
    
    # Calculate rise and fall times for each peak
    # Rise = Time from previous local minimum to Peak
    # Fall = Time from Peak to next local minimum
    
    for p_idx, (pt, pmem) in enumerate(real_peaks):
        # Scan backward for local minimum
        # Local min = lowest point between previous peak (or start) and this peak
        prev_t = real_peaks[p_idx-1][0] if p_idx > 0 else times_us[0]
        
        min_v = pmem
        min_t = pt
        
        # Backward scan is inefficient, let's just use filtered list
        # Optimization: use boolean mask? No, loop is fine for 40k items.
        
        # Find Valley BEFORE Peak
        start_search = 0
        for idx, t in enumerate(times_us):
            if t >= prev_t:
                start_search = idx
                break
                
        # Search from start_search to peak index
        valley_before_mem = pmem
        valley_before_t = pt
        
        for i in range(start_search, len(times_us)):
            if times_us[i] >= pt: break
            if mems[i] < valley_before_mem:
                valley_before_mem = mems[i]
                valley_before_t = times_us[i]
        
        rises.append(pt - valley_before_t)
        
        # Find Valley AFTER Peak
        # Valley is the point with MINIMUM memory between Peak and Next Peak
        next_peak_t = real_peaks[p_idx+1][0] if p_idx < len(real_peaks)-1 else times_us[-1]
        
        # Scan forward from Peak
        min_mem_in_window = pmem
        min_time_in_window = pt
        
        # Optimization: Find start index for peak
        peak_idx = 0
        for idx, t in enumerate(times_us):
            if t >= pt:
                 peak_idx = idx
                 break
                 
        for i in range(peak_idx, len(times_us)):
            t = times_us[i]
            m = mems[i]
            
            if t >= next_peak_t: 
                break # Stop if we hit next peak
            
            # We want the lowest point. 
            # Note: "lowest point" is usually the baseline right before the next rise.
            if m < min_mem_in_window:
                min_mem_in_window = m
                min_time_in_window = t
        
        print(f"Peak {p_idx}: Time={pt}, Mem={pmem}")
        print(f"  Searching range: {pt} -> {next_peak_t}")
        print(f"  Found Min: Time={min_time_in_window}, Mem={min_mem_in_window}")
        print(f"  Fall Duration: {min_time_in_window - pt}")
        
        if min_time_in_window == pt:
             print("  DEBUG: Delta times for first 50 free events after peak:")
             # Find events in valid_events list around the peak
             start_idx = 0
             for idx, e in enumerate(valid_events):
                 if e["time_us"] == pt:
                     start_idx = idx
                     break
             
             free_count = 0
             last_t = pt
             for k in range(start_idx, len(valid_events)):
                 e = valid_events[k]
                 if e["action"] == "free_completed":
                     delta = e["time_us"] - last_t
                     print(f"    Free #{free_count}: Delta={delta}us, Size={e.get('size',0)}")
                     last_t = e["time_us"]
                     free_count += 1
                     if free_count > 50: break

        falls.append(min_time_in_window - pt)
        
    print(f"Rise Durations (us): {rises}")
    print(f"Fall Durations (us): {falls}")
    
    avg_rise = sum(rises)/len(rises)
    avg_fall = sum(falls)/len(falls)
    print(f"Average Ramp Up Time: {avg_rise/1e6:.4f} s")
    print(f"Average Ramp Down Time: {avg_fall/1e6:.4f} s")
    print(f"Ratio (Down / Up): {avg_fall / avg_rise:.2f}")

if __name__ == "__main__":
    analyze_timing(sys.argv[1])
