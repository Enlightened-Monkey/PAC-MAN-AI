import os
import sys
import traceback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def main():
    print("DEBUG: Script started.")
    
    CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'ppo_pacman'))
    print(f"DEBUG: CHECKPOINT_PATH is '{CHECKPOINT_PATH}'")
    print(f"DEBUG: Checking if file exists: {os.path.exists(CHECKPOINT_PATH + '.zip')}")
    
    try:
        print("DEBUG: Importing stable-baselines3 and sb3-contrib...")
        from sb3_contrib import MaskablePPO
        print("DEBUG: Imports successful.")
        
        print("DEBUG: Loading model...")
        # Try loading on CPU first to prevent CUDA crashes!
        model = MaskablePPO.load(CHECKPOINT_PATH, device='cpu')
        print("DEBUG: Model loaded successfully on CPU!")
        print(f"DEBUG: Policy type: {type(model.policy)}")
        
    except BaseException as e:
        print("\nDEBUG: AN EXCEPTION OCCURRED!")
        print(f"DEBUG: Exception type: {type(e)}")
        print(f"DEBUG: Exception message: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
