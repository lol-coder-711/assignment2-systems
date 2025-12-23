
import pickle
import sys
import statistics

def inspect_fall(filepath):
    print(f"Loading {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    events = []
    if "device_traces" in data:
        raw = data["device_traces"]
        for item in raw:
            if isinstance(item, list): events.extend(item)
            else: events.append(item)
    
    # Filter and sort
    valid_events = [e for e in events if isinstance(e, dict) and "time_us" in e]
    valid_events.sort(key=lambda x: x["time_us"])
    
    if not valid_events:
        print("No events found.")
        return

    times = [e["time_us"] for e in valid_events]
    current_mem = 0
    mems = []
    
    # Reconstruct memory curve
    for e in valid_events:
        action = e.get("action", "")
        size = e.get("size", 0)
        if action == "alloc":
            current_mem += size
        elif action == "free_completed":
            current_mem -= size
        mems.append(current_mem)
        
    global_peak = max(mems)
    baseline = min(mems)
    print(f"Global Peak: {global_peak / 1e9:.2f} GB")
    
    # Find the largest "fall" sequence
    # Heuristic: Find a window where we drop from Peak to near Baseline
    
    # Let's look for the point of Global Peak
    peak_idx = mems.index(global_peak)
    peak_time = valid_events[peak_idx]["time_us"]
    print(f"Peak Event Index: {peak_idx} at {peak_time}")
    
    # Scan forward to find when we hit baseline (or close to it)
    end_idx = peak_idx
    for i in range(peak_idx, len(mems)):
        if mems[i] < baseline + (global_peak - baseline) * 0.05: # within 5% of baseline
            end_idx = i
            break
            
    if end_idx == peak_idx:
        print("Could not find a return to baseline after peak.")
        # Try to find just the *next* local minimum
        local_min = global_peak
        for i in range(peak_idx, len(mems)):
            if mems[i] < local_min:
                local_min = mems[i]
                end_idx = i
            # Stop if we start rising significantly again (start of next step)
            if i > peak_idx + 100 and mems[i] > local_min + 0.1 * global_peak:
                break
    
    fall_events = valid_events[peak_idx:end_idx+1]
    start_t = fall_events[0]["time_us"]
    end_t = fall_events[-1]["time_us"]
    duration_us = end_t - start_t
    
    print(f"Fall Phase Found:")
    print(f"  Start Index: {peak_idx}")
    print(f"  End Index:   {end_idx}")
    print(f"  Duration:    {duration_us} us ({duration_us/1000:.2f} ms)")
    print(f"  Total Events in window: {len(fall_events)}")
    
    allocs = [e for e in fall_events if e.get("action") == "alloc"]
    frees = [e for e in fall_events if e.get("action") == "free_completed"]
    kernels = [e for e in fall_events if "cat" in e and e["cat"] == "kernel"] 
    # Note: 'cat' field usually identifies kernels in some traces, or we check if action is missing but name exists?
    # Actually, in PyTorch memory snapshots, 'device_traces' are specifically memory events, but sometimes include context.
    # Let's check typical keys. purely memory snapshot usually ONLY has memory events.
    # But if we enabled NVTX or if this is a timeline trace...
    # The user script used `torch.cuda.memory._record_memory_history`. This primarily records alloc/free.
    # It DOES NOT record kernel execution duration unless linked. 
    # However, if the user sees 'Sawtooth', it's purely memory state.
    
    print(f"  Alloc Events: {len(allocs)}")
    print(f"  Free Events:  {len(frees)}")
    print(f"  Other Events: {len(fall_events) - len(allocs) - len(frees)}")

    
    if len(frees) > 0:
        avg_us_per_free = duration_us / len(frees)
        print(f"  Avg Time per Free (Amortized): {avg_us_per_free:.2f} us")
        print(f"  Frequency: {1e6/avg_us_per_free:.2f} frees/sec")
    
    # Also Check "Rise" Phase
    # Scan backward from peak
    start_rise_idx = peak_idx
    for i in range(peak_idx, -1, -1):
        if mems[i] < baseline + (global_peak - baseline) * 0.05:
            start_rise_idx = i
            break
            
    rise_events = valid_events[start_rise_idx:peak_idx]
    if rise_events:
        r_start_t = rise_events[0]["time_us"]
        r_end_t = rise_events[-1]["time_us"]
        r_dur = r_end_t - r_start_t
        r_allocs = [e for e in rise_events if e.get("action") == "alloc"]
        
        print(f"Rise Phase Found:")
        print(f"  Duration:    {r_dur} us ({r_dur/1000:.2f} ms)")
        print(f"  Alloc Events: {len(r_allocs)}")
        if len(r_allocs) > 0:
            print(f"  Avg Time per Alloc (Amortized): {r_dur / len(r_allocs):.2f} us")

if __name__ == "__main__":
    inspect_fall(sys.argv[1])
