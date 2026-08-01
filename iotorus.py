#!/usr/bin/env python3
import subprocess
import os
from tqdm import tqdm
from datetime import datetime
import re
import sys
import time
import shutil

FILENAME = "iotorus_2_5d"

# Paths
ATHENA_DIR = "/mnt/c/Users/Admin/athena"
BIN = os.path.join(ATHENA_DIR, "bin", "athena")
INPUT = os.path.join(ATHENA_DIR, "inputs", "".join(["athinput.", FILENAME]))

PGENINPUT = "".join([FILENAME, ".cpp"])
PGENINPUT2 = "".join(["--prob=", FILENAME])

BASE_DIR = "outputs"
SHOW_CONSOLE_LOG = False

def parse_tlim(input_file):
    tlim = None
    with open(input_file, "r") as f:
        for line in f:
            if "tlim" in line and "=" in line:
                try:
                    parts = line.strip().split("=")
                    if len(parts) == 2:
                        tlim = float(parts[1].split()[0])
                        break
                except Exception:
                    pass
    if tlim is None:
        raise ValueError("Could not find tlim in input file")
    return tlim

def get_next_program_id(base_dir):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        return 1
    max_id = 0
    for name in os.listdir(base_dir):
        match = re.match(r"Program(\d+)", name)
        if match:
            num = int(match.group(1))
            if num > max_id:
                max_id = num
    return max_id + 1

def create_run_folder(base_dir):
    program_id = get_next_program_id(base_dir)
    timestamp = datetime.now().strftime("%H;%M;%S @ %m-%d-%y")
    run_dir = os.path.join(base_dir, f"Program{program_id} [{timestamp}]")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def archive_files(run_dir):
    shutil.copy(INPUT, os.path.join(run_dir, os.path.basename(INPUT)))
    src_dir = os.path.join(run_dir, "src_snapshot")
    os.makedirs(src_dir, exist_ok=True)
    prob_file = os.path.join(ATHENA_DIR, "src", "pgen", PGENINPUT)
    main_file = os.path.join(ATHENA_DIR, "src", "main.cpp")
    py_file = os.path.join(ATHENA_DIR, "iotorus.py") # Iotorus Python File Name
    anime_file = os.path.join(ATHENA_DIR, "animate_torus_phipps.py") # Iotorus Animation Python File Name
    if os.path.exists(os.path.join(ATHENA_DIR, "src", "pgen", "stable_profile.h")):
        shutil.copy(os.path.join(ATHENA_DIR, "src", "pgen", "stable_profile.h"), src_dir)
    for f in [prob_file, main_file, py_file, anime_file]:
        if os.path.exists(f):
            shutil.copy(f, src_dir)

def run_command(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def build_athena():
    print(">>> NUKING OLD BINARY...")
    run_command(["make", "clean"], cwd=ATHENA_DIR)

    print(">>> CONFIGURING ATHENA++...")
    # Use --cflag to pass -fopenmp to the compiler
    run_command([
        "python3", "configure.py",
        "--prob=iotorus_2_5d",
        "-b",
        "-omp",
        "--nscalars=2",
        "--cflag=-fopenmp"  # Use the correct internal flag
    ], cwd=ATHENA_DIR)

    print(">>> COMPILING...")
    run_command(["make", "-j"], cwd=ATHENA_DIR)


def run_simulation(total_time, run_dir):
    logfile = os.path.join(run_dir, "run.log")
    with open(logfile, "w") as logfile_handle:

        # 1. Define the cores FIRST
        # Leave 1 or 2 cores free so your computer doesn't completely freeze
        max_cores = max(1, os.cpu_count() - 2)
        print(f">>> Launching Athena++ with {max_cores} OpenMP threads...")

        # 2. Build the command using the now-defined max_cores
        # Force the Linux shell to establish the threads before executing the binary
        cmd = [
            "env",
            f"OMP_NUM_THREADS={max_cores}",
            BIN,
            "-i",
            INPUT
        ]

        # 3. Execute without the env=env parameter
        process = subprocess.Popen(
            cmd,
            cwd=run_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # 4. Track the progress
        pbar = tqdm(total=100, desc="Simulation Progress", unit="%")
        for line in process.stdout:
            logfile_handle.write(line)
            logfile_handle.flush()
            if SHOW_CONSOLE_LOG:
                sys.stdout.write(line)
                sys.stdout.flush()

            if "time=" in line:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    current_time = float(time_str)
                    percent = min(int((current_time / total_time) * 100), 100)
                    pbar.n = percent
                    pbar.refresh()
                except Exception:
                    pass

        process.wait()
        pbar.close()

def sound_alarm(repeats=3, delay=0.5):
    try:
        if sys.platform.startswith("win"):
            import winsound
            for _ in range(repeats):
                winsound.Beep(1000, 700)
                time.sleep(delay)
        else:
            for _ in range(repeats):
                print("\a", end="", flush=True)
                time.sleep(delay)
    except Exception:
        pass

if __name__ == "__main__":
    print(">>> Cleaning and building Athena++")
    build_athena()

    print(">>> Parsing tlim from input file")
    TOTAL_TIME = parse_tlim(INPUT)
    print(f">>> Detected tlim = {TOTAL_TIME}")

    run_dir = create_run_folder(BASE_DIR)
    print(f">>> Outputs will be saved in {run_dir}")

    archive_files(run_dir)

    print(">>> Running Io torus simulation (Fresh Start from Stable Data)")
    run_simulation(TOTAL_TIME, run_dir)
    sound_alarm(repeats=3, delay=0.5)