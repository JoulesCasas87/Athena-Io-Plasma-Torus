import os
import sys
import re
import glob
import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter

# --- 1. SETUP PATHS ---
athena_root = "/mnt/c/Users/vince/athena"
athena_vis_path = os.path.join(athena_root, 'vis', 'python')
if athena_vis_path not in sys.path:
    sys.path.insert(0, athena_vis_path)

try:
    import athena_read
except ImportError:
    sys.exit("Error: Could not find 'athena_read.py'. Check your path.")

# --- 2. CONFIG (Sync with Animation) ---
OUTPUT_DT = 0.01
ANGLE_1_USER = 90
R_IO = 5.9
OMEGA_IO = 0.2336
OMEGA_J = 1.0


def auto_detect_event_images(directory):
    """
    Scans the directory for images formatted like 'Day1_12h30m_event.png'
    and calculates the simulation time in hours automatically.
    """
    auto_events = {}
    pattern = re.compile(r"Day(\d+)_(\d+)h(\d+)m", re.IGNORECASE)

    for ext in ('*.png', '*.jpg', '*.jpeg'):
        for file_path in glob.glob(os.path.join(directory, ext)):
            filename = os.path.basename(file_path)

            if "diagnostics" in filename:
                continue

            match = pattern.search(filename)
            if match:
                day = int(match.group(1))
                hrs = int(match.group(2))
                mins = int(match.group(3))

                sim_time_hours = (day - 1) * 24.0 + hrs + (mins / 60.0)

                # Prevent duplicate entries if both .png and .jpg exist for the same time
                if sim_time_hours not in auto_events:
                    auto_events[sim_time_hours] = file_path
                    print(f">>> Detected event image: {filename} at {sim_time_hours:.2f} hrs")

    return dict(sorted(auto_events.items()))


def select_folder():
    root = tk.Tk()
    root.withdraw()
    folder = filedialog.askdirectory(initialdir=os.path.join(athena_root, "outputs"), title="Select Simulation Folder")
    root.destroy()
    return folder


# --- REVERTED TO EXACT ORIGINAL PARSING LOGIC ---
def parse_parameters(sim_dir):
    input_params = {}
    base_input_path = os.path.join(athena_root, "inputs", "athinput.iotorus_2_5d")

    def extract_val(line):
        try:
            key, val = line.split('=', 1)
            key = key.strip()
            val_str = val.split('#')[0].strip()
            if ':' in val_str:
                hrs, mins = val_str.split(':')
                input_params[key] = float(hrs) + float(mins) / 60.0
            else:
                input_params[key] = float(val_str)
        except Exception:
            pass

    if os.path.exists(base_input_path):
        with open(base_input_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'): extract_val(line)

    metadata_path = os.path.join(sim_dir, "render_metadata.txt")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            for line in f:
                if '=' in line and not line.strip().startswith('#'): extract_val(line)

    return {
        'START_IO_PHASE': input_params.get('start_io_phase', 90.0),
        'START_CML': input_params.get('start_cml', 120.0),
        'IOB_CML_MIN': input_params.get('iob_cml_min', 105.0),
        'IOB_CML_MAX': input_params.get('iob_cml_max', 180.0),
        'IOB_PH_MIN': input_params.get('iob_phase_min', 80.0),
        'IOB_PH_MAX': input_params.get('iob_phase_max', 110.0),
        'START_TIME_UTC': input_params.get('start_time_utc', 0.0),
        'OBSERVER_UTC_OFFSET': input_params.get('observer_utc_offset', -5.0),
        'JUP_RISE_HR': input_params.get('jup_rise_local', 18.0),
        'JUP_SET_HR': input_params.get('jup_set_local', 6.0),
        'CODE_TO_HOURS': 9.925 / (2.0 * np.pi)
    }


def mission_clock_format(x, pos):
    day_num = int(x // 24) + 1
    h = int(x % 24)
    rem_m = (x - int(x)) * 60.0
    m = int(rem_m)
    if m == 60: m = 0; h += 1
    if h == 24: h = 0; day_num += 1
    return f"Day {day_num} - {h:02d}:{m:02d}:00"


# ==========================================
# MAIN EXECUTION BLOCK
# ==========================================
if __name__ == '__main__':
    sim_dir = select_folder()
    if not sim_dir:
        sys.exit(">>> Run cancelled. No folder selected.")

    folder_name = os.path.basename(sim_dir)
    print(f">>> Analyzing Data In: {folder_name}")

    EVENT_IMAGES = auto_detect_event_images(sim_dir)

    if not EVENT_IMAGES:
        print(">>> WARNING: No correctly formatted event images found (e.g., Day1_12h30m.png).")

    p = parse_parameters(sim_dir)

    cache_file = os.path.join(sim_dir, "slice_physics_cache_FINAL.csv")

    if not os.path.exists(cache_file):
        print(">>> No cache found. Generating fresh cache from VTK files...")

        vtk_files = sorted([f for f in os.listdir(sim_dir) if f.endswith('.vtk')])
        if not vtk_files:
            sys.exit("Error: No VTK files found in the selected folder, and no cache CSV exists.")

        first_file_path = os.path.join(sim_dir, vtk_files[0])
        x1f, x2f, _, _ = athena_read.vtk(first_file_path)

        phi_centers = 0.5 * (x2f[:-1] + x2f[1:])
        phi_1_rad = np.radians(ANGLE_1_USER % 360)
        idx_phi_1 = np.argmin(np.abs(phi_centers - phi_1_rad))

        r_centers = 0.5 * (x1f[:-1] + x1f[1:])
        idx_r_io = np.argmin(np.abs(r_centers - 5.9))

        time_hist_list = []
        rho_hist_list = []
        va_hist_list = []

        for i, filename in enumerate(vtk_files):
            if i % 100 == 0:
                sys.stdout.write(f"\rReading file {i}/{len(vtk_files)}...")
                sys.stdout.flush()

            filepath = os.path.join(sim_dir, filename)

            try:
                _, _, _, data = athena_read.vtk(filepath)

                bz = 0.0
                for k in ['Bcc', 'bcc', 'B', 'Bcc3', 'bcc3']:
                    if k in data:
                        val = data[k]
                        if k in ['Bcc', 'bcc', 'B'] and val.ndim == 4:
                            bz = val[0, idx_phi_1, idx_r_io, 2]
                        elif val.ndim == 3:
                            bz = val[0, idx_phi_1, idx_r_io]
                        break

                if bz == 0.0:
                    continue

                slice_rho = data['rho'][0, idx_phi_1, idx_r_io]
                slice_vA_phys = np.sqrt((bz ** 2) / (slice_rho + 1e-10)) * 12.57

                with open(filepath, 'rb') as h_f:
                    head = h_f.read(512).decode('utf-8', errors='ignore')
                    match = re.search(r'time=\s*([0-9\.eE\+\-]+)', head)
                    sim_time = float(match.group(1)) if match else i * OUTPUT_DT

                time_hist_list.append(sim_time)
                rho_hist_list.append(slice_rho)
                va_hist_list.append(slice_vA_phys)

            except Exception as e:
                continue

        print("\n>>> Caching complete.")
        results = np.column_stack((time_hist_list, rho_hist_list, va_hist_list))
        np.savetxt(cache_file, results, delimiter=',', header='Time,Density_r6,vA_km_s', comments='')

    else:
        print(">>> Instant loading from safe cache...")

    data = np.loadtxt(cache_file, delimiter=',', skiprows=1)

    time_hist = data[:, 0]
    rho_hist = data[:, 1]
    va_hist = data[:, 2]
    time_hist_hrs = time_hist * p['CODE_TO_HOURS']

    # --- REVERTED TO EXACT ORIGINAL BOOLEAN LOGIC ---
    hist_io_phase = (p['START_IO_PHASE'] + np.degrees(OMEGA_IO * time_hist)) % 360.0
    hist_cml = (p['START_CML'] + np.degrees(OMEGA_J * time_hist)) % 360.0
    earth_time_utc = (time_hist_hrs + p['START_TIME_UTC']) % 24.0
    earth_time_local = (earth_time_utc + p['OBSERVER_UTC_OFFSET']) % 24.0

    if p['JUP_RISE_HR'] > p['JUP_SET_HR']:
        is_visible = (earth_time_local >= p['JUP_RISE_HR']) | (earth_time_local <= p['JUP_SET_HR'])
    else:
        is_visible = (earth_time_local >= p['JUP_RISE_HR']) & (earth_time_local <= p['JUP_SET_HR'])

    iob_condition = ((hist_cml >= p['IOB_CML_MIN']) & (hist_cml <= p['IOB_CML_MAX']) &
                     (hist_io_phase >= p['IOB_PH_MIN']) & (hist_io_phase <= p['IOB_PH_MAX']))

    # =======================================================================
    # PLOTTING: HYBRID LAYOUT (GRAPH + IMAGES)
    # =======================================================================
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, height_ratios=[1.5, 1], hspace=0.3)

    fig.suptitle(f"Io Plasma Torus Diagnostics: {folder_name}\n($r={R_IO} R_J$)", fontsize=16, fontweight='bold',
                 y=0.96)

    ax1 = fig.add_subplot(gs[0, :])
    max_rho = np.max(rho_hist) * 1.1 if len(rho_hist) > 0 else 1.0
    ax1.set_ylim(0, max_rho * 2.8)

    ax1.fill_between(time_hist_hrs, 0, max_rho * 3, where=~is_visible, color='lightgray', alpha=0.6,
                     label='Below Horizon')
    ax1.fill_between(time_hist_hrs, 0, max_rho * 3, where=(iob_condition & is_visible), color='palegreen', alpha=0.8,
                     label='Observable Io-B')

    ax1.plot(time_hist_hrs, rho_hist, color='darkblue', linewidth=2.5, label='Local Density ($cm^{-3}$)')
    ax1.set_ylabel("Density ($cm^{-3}$)", color='darkblue', fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='darkblue')
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    max_va = np.max(va_hist)
    ax1_twin.set_ylim(0, max_va * 1.5 if max_va > 0 else 100)
    ax1_twin.plot(time_hist_hrs, va_hist, color='cyan', linewidth=1.5, label='Alfvén Speed (km/s)')
    ax1_twin.set_ylabel("Alfvén Speed (km/s)", color='teal', fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor='teal')

    ax1.set_xlabel("Simulation Time (Mission Clock)", fontweight='bold', fontsize=12)
    ax1.xaxis.set_major_formatter(FuncFormatter(mission_clock_format))
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
    ax1.set_xlim(0, time_hist_hrs[-1] if len(time_hist_hrs) > 0 else 0.1)

    event_times = list(EVENT_IMAGES.keys())
    colors = ['magenta', 'orange']

    for idx, event_time in enumerate(event_times):
        c = colors[idx % len(colors)]
        formatted_event_time = mission_clock_format(event_time, None)
        label_text = f"Event {idx + 1}: {formatted_event_time}"

        ax1.axvline(x=event_time, color=c, linestyle='--', linewidth=2, zorder=5)
        ax1.text(event_time + 1, max_rho * 2.5, label_text, color=c, fontweight='bold', fontsize=10, rotation=90,
                 va='top')
        ax1.plot([], [], color=c, linestyle='--', linewidth=2, label=label_text)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_1t, labels_1t = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_1t, labels_1 + labels_1t, loc='upper left', fontsize=9, ncol=2)

    image_axes = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]

    for idx, (event_time, img_path) in enumerate(EVENT_IMAGES.items()):
        if idx >= 2: break
        ax_img = image_axes[idx]
        try:
            img = mpimg.imread(img_path)
            ax_img.imshow(img)
            formatted_time = mission_clock_format(event_time, None)
            ax_img.set_title(f"Event {idx + 1} ({formatted_time})", fontweight='bold')
            ax_img.axis('off')
        except FileNotFoundError:
            ax_img.text(0.5, 0.5, f"Image not found:\n{os.path.basename(img_path)}",
                        ha='center', va='center', fontsize=10, color='red', transform=ax_img.transAxes)
            ax_img.set_title(f"Event {idx + 1} Placeholder", fontweight='bold')
            ax_img.axis('off')

    plot_path = os.path.join(sim_dir, f"diagnostics_summary_{folder_name}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f">>> Plot saved to: {plot_path}")

    plt.show()