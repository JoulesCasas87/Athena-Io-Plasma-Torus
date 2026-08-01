#!/usr/bin/env python3
import subprocess
import os
from tqdm import tqdm
from datetime import datetime
import re
import sys
import time
import shutil
import threading

FILENAME = "iotorus_parallel"

# Paths
ATHENA_DIR = "/mnt/c/Users/Admin/athena"
BIN = os.path.join(ATHENA_DIR, "bin", "athena")
INPUT = os.path.join(ATHENA_DIR, "inputs", "".join(["athinput.", FILENAME]))

PGENINPUT = "".join([FILENAME, ".cpp"])
PGENINPUT2 = "".join(["--prob=", FILENAME])

BASE_DIR = os.path.join(ATHENA_DIR, "outputs")

# --- SIMULATION BATCH CONFIGURATION ---
# Define your simultaneous runs and their parameter overrides
SIMULATIONS = [
    {
        "name": "K4LED_May_23_2019",
        "overrides": [
            "problem/start_io_phase=38.97",
            "problem/start_cml=240.49",
            "problem/observer_utc_offset=-5.0",
            "problem/jup_rise_local=21.19",
            "problem/jup_set_local=6.44"
        ]
    },
    {
        "name": "K4LED_August_05_2018",
        "overrides": [
            "problem/start_io_phase=259.97",
            "problem/start_cml=27.37",
            "problem/observer_utc_offset=-4.0",
            "problem/jup_rise_local=13.19",
            "problem/jup_set_local=23.43"
        ]
    },
    {
        "name": "K4LED_May_04_2018",
        "overrides": [
            "problem/start_io_phase=51.68",
            "problem/start_cml=67.14",
            "problem/observer_utc_offset=-5.0"
            "problem/jup_rise_local=19.53",
            "problem/jup_set_local=6.05"
        ]
    },
    {
        "name": "K4LED_July_29_2018",
        "overrides": [
            "problem/start_io_phase=71.95",
            "problem/start_cml=264.2",
            "problem/observer_utc_offset=-4.0",
            "problem/jup_rise_local=13.45",
            "problem/jup_set_local=00.10"
        ]
    },
    {
        "name": "Aguirre_January_25_2026",
        "overrides": [
            "problem/start_io_phase=61.50",
            "problem/start_cml=351.8",
            "problem/observer_utc_offset=-7.0",
            "problem/jup_rise_local=16.27",
            "problem/jup_set_local=6.16"
        ]
    },
    {
        "name": "Aguirre_February_01_2026",
        "overrides": [
            "problem/start_io_phase=46.64",
            "problem/start_cml=326.17",
            "problem/observer_utc_offset=-7.0",
            "problem/jup_rise_local=15.56",
            "problem/jup_set_local=5.45"
        ]
    },
    {
        "name": "Aguirre_November_15_2025",
        "overrides": [
            "problem/start_io_phase=6.34",
            "problem/start_cml=94.83",
            "problem/observer_utc_offset=-7.0",
            "problem/jup_rise_local=21.43",
            "problem/jup_set_local=11.21"
        ]
    },
    {
        "name": "Aguirre_October_25_2023",
        "overrides": [
            "problem/start_io_phase=57.21",
            "problem/start_cml=8.51",
            "problem/observer_utc_offset=-7.0",
            "problem/jup_rise_local=18.18",
            "problem/jup_set_local=7.19"
        ]
    },
    {
        "name": "HNRAO_November_01_2023",
        "overrides": [
            "problem/start_io_phase=42.50",
            "problem/start_cml=343.25",
            "problem/observer_utc_offset=-4.0",
            "problem/jup_rise_local=18.28",
            "problem/jup_set_local=8.07"
        ]
    },
    {
        "name": "HNRAO_November_08_2023",
        "overrides": [
            "problem/start_io_phase=27.79",
            "problem/start_cml=317.92",
            "problem/observer_utc_offset=-4.0",
            "problem/jup_rise_local=17.58",
            "problem/jup_set_local=07.34"
        ]
    }
]


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
    py_file = os.path.join(ATHENA_DIR, "iotorus.py")
    anime_file = os.path.join(ATHENA_DIR, "animate_torus_phipps.py")

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
    run_command([
        "python3", "configure.py",
        PGENINPUT2,
        "--coord=cylindrical",
        "-b",
        "-omp",
        "--nscalars=2",
        "--cflag=-fopenmp"
    ], cwd=ATHENA_DIR)

    print(">>> COMPILING...")
    run_command(["make", "-j"], cwd=ATHENA_DIR)


# --- THREADED SIMULATION FUNCTION ---
def run_single_simulation(sim_config, parent_dir, base_tlim, threads, bar_position):
    sim_name = sim_config["name"]
    overrides = sim_config["overrides"]

    # Create the sub-directory for this specific run
    sim_dir = os.path.join(parent_dir, sim_name)
    os.makedirs(sim_dir, exist_ok=True)

    logfile = os.path.join(sim_dir, "run.log")

    # Check if this specific override changes the tlim
    sim_tlim = base_tlim
    for ovr in overrides:
        if "tlim=" in ovr:
            sim_tlim = float(ovr.split("=")[1])

    # --- CREATE THE SPECIAL METADATA FILE FOR THE RENDERER ---
    metadata_path = os.path.join(sim_dir, "render_metadata.txt")
    with open(metadata_path, "w") as meta_file:
        meta_file.write("# Explicit overrides injected by batch script\n")
        for ovr in overrides:
            # Strip 'problem/' or 'time/' from the string so it's just 'key=value'
            if "=" in ovr and "/" in ovr:
                clean_key_val = ovr.split("/", 1)[1]
                meta_file.write(f"{clean_key_val}\n")

    with open(logfile, "w") as logfile_handle:
        cmd = [
                  "env",
                  f"OMP_NUM_THREADS={threads}",
                  BIN,
                  "-i",
                  INPUT,
                  "-d",
                  sim_dir
              ] + overrides

        process = subprocess.Popen(
            cmd,
            cwd=sim_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        # Initialize the specific progress bar for this thread
        pbar = tqdm(total=100, desc=f"{sim_name[:15]:<15}", position=bar_position, unit="%", leave=True)

        for line in process.stdout:
            logfile_handle.write(line)
            logfile_handle.flush()

            if "time=" in line:
                try:
                    time_str = line.split("time=")[1].split()[0]
                    current_time = float(time_str)
                    percent = min(int((current_time / sim_tlim) * 100), 100)
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

    print(">>> Parsing base tlim from input file")
    BASE_TLIM = parse_tlim(INPUT)
    print(f">>> Detected base tlim = {BASE_TLIM}")

    # Create the master parent folder (e.g., Program70 [Time])
    parent_run_dir = create_run_folder(BASE_DIR)
    print(f">>> Master batch folder created at: {parent_run_dir}")

    # Archive the snapshot into the master folder
    archive_files(parent_run_dir)

    # Calculate safe CPU thread limits per simulation
    total_cores_to_use = max(1, os.cpu_count() - 2)
    threads_per_sim = max(1, total_cores_to_use // len(SIMULATIONS))

    print(f"\n>>> Launching {len(SIMULATIONS)} simultaneous simulations.")
    print(f">>> Allocating {threads_per_sim} OpenMP threads per simulation (Total Cores: {total_cores_to_use}).\n")

    # Launch threads
    threads = []
    for idx, sim in enumerate(SIMULATIONS):
        t = threading.Thread(
            target=run_single_simulation,
            args=(sim, parent_run_dir, BASE_TLIM, threads_per_sim, idx)
        )
        t.start()
        threads.append(t)
        time.sleep(0.5)  # Stagger launches to prevent IO traffic jams

    # Wait for all simulations to finish
    for t in threads:
        t.join()

    # Move cursor past the progress bars before final print
    print("\n" * len(SIMULATIONS))
    print(">>> ALL SIMULATIONS COMPLETE!")
    sound_alarm(repeats=3, delay=0.5)