import pickle
import sys

def debug(filepath):
    print(f"DEBUG: {filepath}")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    
    if "device_traces" in data:
        traces = data["device_traces"]
        print(f"Total events: {len(traces)}")
        for i, e in enumerate(traces[:5]):
            print(f"Event {i}: type={type(e)} keys={e.keys() if isinstance(e, dict) else 'N/A'}")
            if isinstance(e, dict):
                print(f"  Content: {e}")
    else:
        print("No device_traces found.")

if __name__ == "__main__":
    debug(sys.argv[1])
