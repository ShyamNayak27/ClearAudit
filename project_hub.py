import os
import sys
import subprocess
import time
import socket

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def main():
    # Anchor to the directory of the script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    clear_screen()
    print("="*60)
    print("        VIETNAM FRAUD DETECTOR - PROJECT CONTROL HUB")
    print("="*60)

    processes = []
    
    # 0. Start Main Landing Page (Primary Entry Point)
    if not is_port_in_use(8081):
        print("[*] Starting Main Portal (Port 8081)...")
        p = subprocess.Popen(["npx", "http-server", "frontend/main_landing_page", "-p", "8081", "--cors"],
                             cwd=base_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    else:
        print("[!] Main Portal is already running on Port 8081.")

    # 1. Start Backend API
    if not is_port_in_use(8000):
        print("[*] Starting Backend API (Port 8000)...")
        p = subprocess.Popen(["uvicorn", "src.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
                             cwd=base_dir, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    else:
        print("[!] Backend API is already running on Port 8000.")

    # 2. Start Technical Page Frontend
    if not is_port_in_use(8082):
        print("[*] Starting Technical Page Frontend (Port 8082)...")
        # Ensure we serve the internal directory so index.html is at root
        p = subprocess.Popen(["npx", "http-server", "frontend/technical_page", "-p", "8082", "--cors"],
                             cwd=base_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    else:
        print("[!] Technical Page Frontend is already running on Port 8082.")

    # 3. Start Service Page Frontend
    if not is_port_in_use(8083):
        print("[*] Starting Service Page Frontend (Port 8083)...")
        p = subprocess.Popen(["npx", "http-server", "frontend/service_page", "-p", "8083", "--cors"],
                             cwd=base_dir, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
    else:
        print("[!] Service Page Frontend is already running on Port 8083.")

    print("\n[ACTIVE SERVICES & LINKS]")
    print(f"{'-'*30}")
    
    # Wait a moment for all servers to bind
    time.sleep(1)
    
    # Ready checks
    results = []
    results.append(("MAIN PORTAL (Entry)      ", "http://127.0.0.1:8081", is_port_in_use(8081)))
    results.append(("Backend API (FastAPI)     ", "http://127.0.0.1:8000", is_port_in_use(8000)))
    results.append(("Technical Page (Portfolio)", "http://127.0.0.1:8082", is_port_in_use(8082)))
    results.append(("Service Page (Commercial) ", "http://127.0.0.1:8083", is_port_in_use(8083)))

    for label, link, active in results:
        status = "[ONLINE]" if active else "[STARTING...]"
        print(f"{label} : {link} {status}")
    
    print(f"{'-'*30}")
    print("\n[OPERATIONAL COMMANDS]")
    print(f"{'-'*30}")
    print("B. MODEL TRAINING:")
    print("   - Train XGBoost Supervised : python src/models/train_supervised.py")
    print("   - Train Autoencoder (AE)   : python src/models/train_autoencoder.py")
    print("   - Optimize AE Threshold    : python src/models/epoch_sweep.py")
    print("\nC. DATA & VERIFICATION:")
    print("   - Run PaySim Simulation    : python src/data_gen/simulate_paysim_vn.py")
    print("   - Run Model Evaluation     : python src/models/evaluate.py")
    print("   - Batch Verification       : python verify_2_1_2_2.py")
    print(f"{'-'*30}")
    print("\n[NOTE] Idea 1 and Idea 2 have been disabled from localhost hosting.")
    print("="*60)
    print("\nPress Ctrl+C to shutdown all services and exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n[*] Shutting down all services...")
        for p in processes:
            p.terminate()
        print("[*] Done. Goodbye!")

if __name__ == "__main__":
    main()
