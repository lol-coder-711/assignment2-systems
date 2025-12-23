import pickle
import sys
import os

def analyze(filepath):
    print(f"Analyzing {os.path.basename(filepath)}...")
    if not os.path.exists(filepath):
        print("File not found.")
        return

    try:
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        
        if isinstance(data, dict) and "device_traces" in data:
            current_mem = 0
            peak_mem = 0
            max_alloc_size = 0
            
            # Flatten traces if list of lists
            all_events = []
            raw_traces = data["device_traces"]
            for item in raw_traces:
                if isinstance(item, list):
                    all_events.extend(item)
                else:
                    all_events.append(item)
            
            # Filter and Sort by time
            events = []
            for e in all_events:
                 if isinstance(e, dict) and "action" in e and "time_us" in e:
                     events.append(e)
            
            events.sort(key=lambda x: x["time_us"])
                    
            # Filter for memory events
            count = 0
            for e in events:
                if "action" not in e: continue
                
                size = e.get("size", 0)
                if e["action"] == "alloc":
                    current_mem += size
                    count += 1
                    if size > max_alloc_size:
                        max_alloc_size = size
                elif e["action"] == "free_completed":
                    current_mem -= size
                
                if current_mem > peak_mem:
                    peak_mem = current_mem
            
            print(f"  Processed {count} alloc events.")
            print(f"  Peak Memory: {peak_mem / 1024**3:.4f} GB")
            print(f"  Max Alloc: {max_alloc_size / 1024**2:.2f} MB")

        elif isinstance(data, list):
             # Try to handle list format if present
             pass

    except Exception as e:
        print(f"  Error: {e}")

if __name__ == "__main__":
    for f in sys.argv[1:]:
        analyze(f)
