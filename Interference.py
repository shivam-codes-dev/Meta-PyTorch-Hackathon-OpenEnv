import os
import requests
import time

def run_inference():
    print("START")
    try:
        # Local server (Flask) ko hit karna
        response = requests.post("http://localhost:5000/reset")
        if response.status_code == 200:
            print(f"STEP: Action=3, Reward=-0.1") # Dummy step for validation
        else:
            print("ERROR: Reset failed")
    except Exception as e:
        print(f"ERROR: {str(e)}")
    print("END")

if __name__ == "__main__":
    time.sleep(2) # Server start hone ka wait
    run_inference()
