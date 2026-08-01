import sys
import os
import time
import re
import subprocess
import shutil
from multiprocessing import Pool, cpu_count
import numpy as np
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, ListedColormap
from matplotlib.ticker import FuncFormatter

# --- 1. SETUP PATHS ---
athena_vis_path = os.path.join(os.getcwd(), 'vis', 'python')
if athena_vis_path not in sys.path:
    sys.path.insert(0, athena_vis_path)

try:
    import athena_read
except ImportError:
    sys.exit("Error: Could not find 'athena_read.py'. Check your path.")

# --- AUTO-DETECT LATEST FOLDER ---
base_outputs_dir = "/mnt/c/Users/Admin/athena/outputs/"
subfolders = [f.path for f in os.scandir(base_outputs_dir) if f.is_dir()]
data_dir = max(subfolders, key=os.path.getmtime)
print(f"Automatically selected latest folder: {os.path.basename(data_dir)}")

all_files = os.listdir(data_dir)
vtk_files = sorted([f for f in all_files if f.endswith('.vtk')])

# --- 2. CONFIGURATION & COORDINATE MATH ---
OUTPUT_DT = 0.00879  # MATCH WITH DT IN INPUT FILE!!!!!!!!!!
ANGLE_1_USER = 90
ANGLE_2_USER = 240
R_IO = 5.9
OMEGA_IO = 0.2336
OMEGA_J = 1.0

# Rotate the visual "camera" 90 degrees so 0 is Top and 180 is Bottom (Earth)
PLOT_OFFSET = np.pi / 2.0

# --- PARSE INPUT FILE FOR CUSTOM VARIABLES ---
# Hardcode the root Athena directory so Python doesn't look in the outputs folder
athena_root = "/mnt/c/Users/Admin/athena/inputs"
input_file_path = os.path.join(athena_root, "athinput.iotorus_2_5d")

input_params = {}
if os.path.exists(input_file_path):
    print(f">>> Successfully located input file at: {input_file_path}")
    with open(input_file_path, 'r') as f:
        for line in f:
            if '=' in line and not line.strip().startswith('#'):
                key, val = line.split('=', 1)
                val_str = val.split('#')[0].strip()

                # Automatically convert HH:MM to decimal hours
                if ':' in val_str:
                    hrs, mins = val_str.split(':')
                    input_params[key.strip()] = float(hrs) + float(mins) / 60.0
                else:
                    try:
                        input_params[key.strip()] = float(val_str)
                    except ValueError:
                        pass
else:
    print(f"!!! WARNING: Could not find input file at {input_file_path}.")
    print("!!! Falling back to default 90-degree values.")

START_IO_PHASE = input_params.get('start_io_phase', 90.0)
START_CML = input_params.get('start_cml', 120.0)
IOB_CML_MIN = input_params.get('iob_cml_min', 105.0)
IOB_CML_MAX = input_params.get('iob_cml_max', 180.0)
IOB_PH_MIN = input_params.get('iob_phase_min', 80.0)
IOB_PH_MAX = input_params.get('iob_phase_max', 110.0)

START_TIME_UTC = input_params.get('start_time_utc', 0.0)
OBSERVER_UTC_OFFSET = input_params.get('observer_utc_offset', -5.0)
JUP_RISE_LOCAL = input_params.get('jup_rise_local', 18.0)
JUP_SET_LOCAL = input_params.get('jup_set_local', 6.0)

CODE_TO_HOURS = 1.0

print(f">>> Parsed Start Phase: {START_IO_PHASE}°, Start CML: {START_CML}°")

def user_to_math_rad(user_deg):
    return np.radians(user_deg % 360)


phi_1_rad = user_to_math_rad(ANGLE_1_USER)
phi_2_rad = user_to_math_rad(ANGLE_2_USER)

first_file_path = os.path.join(data_dir, vtk_files[0])
x1f, x2f, x3f, data0 = athena_read.vtk(first_file_path)

r_centers = 0.5 * (x1f[:-1] + x1f[1:])
phi_centers = 0.5 * (x2f[:-1] + x2f[1:])

# Create 2D arrays with the camera offset applied
R_centers_2d, PHI_centers_2d = np.meshgrid(r_centers, phi_centers)
X_centers = R_centers_2d * np.cos(PHI_centers_2d + PLOT_OFFSET)
Y_centers = R_centers_2d * np.sin(PHI_centers_2d + PLOT_OFFSET)

idx_phi_1 = np.argmin(np.abs(phi_centers - phi_1_rad))
idx_phi_2 = np.argmin(np.abs(phi_centers - phi_2_rad))

R_faces, PHI_faces = np.meshgrid(x1f, x2f)
X_faces = R_faces * np.cos(PHI_faces + PLOT_OFFSET)
Y_faces = R_faces * np.sin(PHI_faces + PLOT_OFFSET)

vmin, vmax = 1, 1e5
r_max = x1f[-1]

frames_dir = os.path.join(data_dir, "temp_frames")
os.makedirs(frames_dir, exist_ok=True)


# --- EMPIRICAL TARGET LINE (Phipps & Withers 2017) ---
def get_phipps_target(r_array):
    N1, C1, W1 = 1710.0, 5.23, 0.20
    N2, C2, W2 = 2180.0, 5.60, 0.08
    N3, C3, W3 = 2160.0, 5.89, 0.32
    N4, C4, W4 = 1601.0, 5.52, 1.88

    n_val = np.zeros_like(r_array)
    scale_factor = 10000.0 / 3000.0

    for idx, r in enumerate(r_array):
        if r < 4.8:
            n_val[idx] = 3.0 / scale_factor
        elif r < 6.1:
            n_val[idx] = (N1 * np.exp(-((r - C1) / W1) ** 2) +
                          N2 * np.exp(-((r - C2) / W2) ** 2) +
                          N3 * np.exp(-((r - C3) / W3) ** 2))
        else:
            n_val[idx] = N4 * np.exp(-((r - C4) / W4) ** 2)

    return n_val * scale_factor


target_density_curve = get_phipps_target(r_centers)


def get_sim_time(filepath, frame_index):
    try:
        with open(filepath, 'rb') as f:
            header = f.read(512).decode('utf-8', errors='ignore')
            match = re.search(r'time=\s*([0-9\.eE\+\-]+)', header, re.IGNORECASE)
            if match: return float(match.group(1))
    except:
        pass
    return float(frame_index * OUTPUT_DT)


# --- 4. THE PARALLEL WORKER FUNCTION ---
def render_frame(args):
    # Unpack the historical arrays passed from the pre-calculation step
    frame_index, filename, time_hist, rho_hist = args
    filepath = os.path.join(data_dir, filename)

    # Define two separate output paths
    out_img_clean = os.path.join(frames_dir, f"clean_{frame_index:05d}.png")
    out_img_vec = os.path.join(frames_dir, f"vec_{frame_index:05d}.png")

    if os.path.exists(out_img_clean) and os.path.exists(out_img_vec): return

    _, _, _, data = athena_read.vtk(filepath)
    rho_new = data['rho'][0, :, :]

    press_new = data.get('press', np.zeros_like(rho_new))[0, :, :]

    # 1. Vacuum Safety Floor
    rho_safe = np.where(rho_new < 1e-3, 1e-3, rho_new)
    temp_code = np.divide(press_new, rho_safe)

    # 2. Physics Conversion to eV
    CONVERSION_FACTOR_EV = 32.7
    temp_eV = temp_code * CONVERSION_FACTOR_EV

    v_r_new = data['vel'][0, :, :, 0]
    v_phi_new = data['vel'][0, :, :, 1]

    sim_time = get_sim_time(filepath, frame_index)

    # --- TRACER BLOB EXTRACTION ---
    r0_frac = data.get('r0', np.zeros_like(rho_new))[0, :, :]
    tracer_rho = r0_frac * rho_new

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Io Plasma Torus & Thermal Profile", fontsize=18)
    plt.subplots_adjust(hspace=0.3)

    ax_2d = axs[0, 0]
    ax_1d_avg = axs[0, 1]
    ax_1d_a1 = axs[1, 0]
    ax_1d_a2 = axs[1, 1]

    # --- 2.5D View (Base Density) ---
    ax_2d.set_aspect('equal')
    ax_2d.set_facecolor('black')
    ax_2d.set_title("Equatorial Plane (Density)")
    pcm = ax_2d.pcolormesh(X_faces, Y_faces, rho_new, shading='flat', cmap='plasma', norm=LogNorm(vmin=vmin, vmax=vmax))
    fig.colorbar(pcm, ax=ax_2d, fraction=0.046, pad=0.04, label='Density')

    # --- BLOB OUTLINES (CONTOURS) ---
    contour_levels = [3000, 5000]
    CS = ax_2d.contour(X_centers, Y_centers, rho_new, levels=contour_levels,
                       colors=['lime', 'white'], linewidths=[2, 2],
                       linestyles='solid', zorder=12)
    ax_2d.clabel(CS, CS.levels, inline=True, fontsize=10, colors=['lime', 'white'])

    # --- TRACER BLOB VISUALIZATION ---
    if np.max(tracer_rho) > 0.1:
        CS_tracer = ax_2d.contour(X_centers, Y_centers, tracer_rho,
                                  levels=[0.5, 2.0, 5.0],
                                  colors=['purple'], linewidths=[2, 3, 4],
                                  linestyles='solid', zorder=13)
        ax_2d.clabel(CS_tracer, inline=True, fontsize=10, colors=['cyan'])

    # --- IO POSITION & ORBIT ---
    phi_io = (np.radians(START_IO_PHASE) + OMEGA_IO * sim_time) % (2.0 * np.pi)
    idx_phi_io = np.argmin(np.abs(phi_centers - phi_io))
    x_io = R_IO * np.cos(phi_io + PLOT_OFFSET)
    y_io = R_IO * np.sin(phi_io + PLOT_OFFSET)

    ax_2d.add_patch(plt.Circle((0, 0), 1.0, color='gray', label='Jupiter'))
    ax_2d.plot([x_io], [y_io], 'wo', markersize=6, markeredgecolor='black', label="Io Orbit", zorder=15)

    # --- VISIBLE LOCAL SLICE LINES ---
    ax_2d.plot([0, r_max * np.cos(phi_1_rad + PLOT_OFFSET)], [0, r_max * np.sin(phi_1_rad + PLOT_OFFSET)],
               color='white', linestyle='--', linewidth=3, zorder=10,
               label=f'{ANGLE_1_USER}° Cut')
    ax_2d.plot([0, r_max * np.cos(phi_2_rad + PLOT_OFFSET)], [0, r_max * np.sin(phi_2_rad + PLOT_OFFSET)],
               color='lightgray', linestyle=':', linewidth=3, zorder=10,
               label=f'{ANGLE_2_USER}° Cut')

    # --- MAGNETIC TILT VISUALIZATION ---
    phi_mag = (np.radians(START_CML) + OMEGA_J * sim_time) % (2.0 * np.pi)
    ax_2d.plot([-r_max * np.cos(phi_mag + PLOT_OFFSET), r_max * np.cos(phi_mag + PLOT_OFFSET)],
               [-r_max * np.sin(phi_mag + PLOT_OFFSET), r_max * np.sin(phi_mag + PLOT_OFFSET)],
               color='cyan', linestyle='--', linewidth=1.5, alpha=0.7,
               label='Mag. Equator (Peak Inj.)')

    phi_min = phi_mag + (np.pi / 2.0)
    ax_2d.plot([-r_max * np.cos(phi_min + PLOT_OFFSET), r_max * np.cos(phi_min + PLOT_OFFSET)],
               [-r_max * np.sin(phi_min + PLOT_OFFSET), r_max * np.sin(phi_min + PLOT_OFFSET)],
               color='magenta', linestyle=':', linewidth=1.5, alpha=0.7,
               label='Max Mag. Lat (Min Inj.)')

    # --- INJECTION MATH & TEXT OVERLAYS ---
    omega_rel = OMEGA_J - OMEGA_IO
    tilt_wave = np.power(np.cos(omega_rel * sim_time), 2)
    mdot_io_live = 25000.0 + (25000.0 - 25000.0) * tilt_wave

    real_hours = sim_time * CODE_TO_HOURS
    phi_io_deg = np.degrees(phi_io)

    # Convert Elapsed Time to Day - HH:MM:SS
    sim_day = int(real_hours // 24) + 1
    sim_hh = int(real_hours % 24)
    rem_m = (real_hours - int(real_hours)) * 60.0
    sim_mm = int(rem_m)
    sim_ss = int(round((rem_m - sim_mm) * 60.0))

    # Calculate Live UTC Time
    sim_time_utc = (real_hours + START_TIME_UTC) % 24.0
    utc_hh = int(sim_time_utc)
    utc_mm = int(round((sim_time_utc - utc_hh) * 60))
    if utc_mm == 60:
        utc_hh = (utc_hh + 1) % 24
        utc_mm = 0

    # Calculate Local Observer Time
    sim_time_local = (sim_time_utc + OBSERVER_UTC_OFFSET) % 24.0
    loc_hh = int(sim_time_local)
    loc_mm = int(round((sim_time_local - loc_hh) * 60))
    if loc_mm == 60:
        loc_hh = (loc_hh + 1) % 24
        loc_mm = 0

    ax_2d.text(0.05, 0.95, f"Elapsed: Day {sim_day} - {sim_hh:02d}:{sim_mm:02d}:{sim_ss:02d}",
               transform=ax_2d.transAxes, color='white', fontsize=12)
    ax_2d.text(0.05, 0.90, f"Obs Time (UTC): {utc_hh:02d}:{utc_mm:02d}", transform=ax_2d.transAxes, color='orange',
               fontsize=12)
    #ax_2d.text(0.05, 0.85, f"Local Time (GA): {loc_hh:02d}:{loc_mm:02d}", transform=ax_2d.transAxes, color='cyan',
               #fontsize=12)
    ax_2d.text(0.05, 0.85, f"Io Phase: {phi_io_deg:.1f}°", transform=ax_2d.transAxes, color='yellow', fontsize=12)
    ax_2d.text(0.05, 0.80, f"Injection: {mdot_io_live:.0f}", transform=ax_2d.transAxes, color='cyan', fontsize=12)

    # --- 1D Graphs ---
    def setup_1d_graph(ax, title, y_data):
        ax.set_title(title)
        ax.set_xlabel(r'Radial Distance ($R_J$)')
        ax.set_ylabel(r'Density ($\rho$)')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.2)
        ax.set_ylim(vmin, vmax)
        ax.set_xlim(x1f[0], x1f[-1])
        ax.plot(r_centers, y_data, color='blue', linewidth=2, label='Simulation Density')
        ax.plot(r_centers, target_density_curve, color='green', linestyle='--', linewidth=1.5, alpha=0.8,
                label='Target Profile')
        ax.axvline(x=R_IO, color='red', linestyle='--', alpha=0.5)

    def setup_time_series_graph(ax, current_time, time_hist, rho_hist):
        time_hist_hrs = time_hist * CODE_TO_HOURS
        current_time_hrs = current_time * CODE_TO_HOURS

        ax.set_title(f"Average Density at {ANGLE_1_USER}° Slice vs Time")
        ax.set_xlabel('Simulation Time (HH:MM)')
        ax.set_ylabel('Average Density')
        ax.set_yscale('linear')
        ax.grid(True, which="both", ls="-", alpha=0.2)

        # --- Format X-Axis ticks to Day X - HH:MM:SS ---
        def mission_clock_format(x, pos):
            # Calculate absolute days (assuming T=0 starts on Day 1)
            day_num = int(x // 24) + 1

            # Calculate remaining hours within that day
            h = int(x % 24)

            # Calculate minutes and seconds from the decimal remainder
            rem_m = (x - int(x)) * 60.0
            m = int(rem_m)

            rem_s = (rem_m - m) * 60.0
            s = int(round(rem_s))

            # Catch rounding overflows
            if s == 60:
                s = 0
                m += 1
            if m == 60:
                m = 0
                h += 1
            if h == 24:
                h = 0
                day_num += 1

            return f"Day {day_num} - {h:02d}:{m:02d}:{s:02d}"

        ax.xaxis.set_major_formatter(FuncFormatter(mission_clock_format))

        # To prevent the labels from overlapping due to their new length, slightly rotate them
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')

        # --- LOCK AXES & DEFINE MAX_RHO ---
        max_time_hrs = time_hist_hrs[-1] if len(time_hist_hrs) > 0 else 0.1
        max_rho = np.max(rho_hist) * 1.1 if len(rho_hist) > 0 else 1.0
        ax.set_xlim(0, max(max_time_hrs, 0.1))
        ax.set_ylim(0, max(max_rho, 1.0))

        # --- 1. CALCULATE CML & IO PHASE ---
        hist_io_phase = (START_IO_PHASE + np.degrees(OMEGA_IO * time_hist)) % 360.0
        hist_cml = (START_CML + np.degrees(OMEGA_J * time_hist)) % 360.0

        iob_condition = ((hist_cml >= IOB_CML_MIN) & (hist_cml <= IOB_CML_MAX) &
                         (hist_io_phase >= IOB_PH_MIN) & (hist_io_phase <= IOB_PH_MAX))

        # --- 2. CALCULATE EARTH VISIBILITY CYCLE ---
        # The main timeline runs in UTC to match the RadioJOVE data files
        earth_time_utc = (time_hist_hrs + START_TIME_UTC) % 24.0

        # Shift the UTC time to the observer's local time for horizon calculations
        earth_time_local = (earth_time_utc + OBSERVER_UTC_OFFSET) % 24.0

        if JUP_RISE_LOCAL > JUP_SET_LOCAL:  # Crosses midnight
            is_visible = (earth_time_local >= JUP_RISE_LOCAL) | (earth_time_local <= JUP_SET_LOCAL)
        else:  # Daytime observation
            is_visible = (earth_time_local >= JUP_RISE_LOCAL) & (earth_time_local <= JUP_SET_LOCAL)
        # --- 3. APPLY SHADING MASKS ---
        # Shade gray when Jupiter is below the horizon / daytime
        ax.fill_between(time_hist_hrs, 0, max_rho, where=~is_visible,
                        color='gray', alpha=0.3, label='Below Horizon')

        # Highlight green ONLY when Io-B conditions are met AND Jupiter is visible
        ax.fill_between(time_hist_hrs, 0, max_rho, where=(iob_condition & is_visible),
                        color='lime', alpha=0.4, label='Observable Io-B')

        # Mask the density line to animate over time
        mask_current = time_hist_hrs <= current_time_hrs
        plot_t = time_hist_hrs[mask_current]
        plot_rho = rho_hist[mask_current]

        ax.plot(plot_t, plot_rho, color='yellow', linewidth=2, label='90° Slice Avg')

        if len(plot_t) > 0:
            ax.plot(plot_t[-1], plot_rho[-1], 'ro', markersize=8, zorder=10)

        ax.legend(loc='upper right', fontsize=8)

    # EXECUTE THE GRAPHS
    setup_1d_graph(ax_1d_avg, "Global Azimuthal Average (With Temperature)", np.mean(rho_new, axis=0))
    setup_1d_graph(ax_1d_a1, f"Local Profile at {ANGLE_1_USER}°", rho_new[idx_phi_1, :])

    # NEW CALL: Pass the historical data to the time-series setup
    setup_time_series_graph(ax_1d_a2, sim_time, time_hist, rho_hist)

    ax_temp = ax_1d_avg.twinx()
    ax_temp.set_ylabel('Temperature (eV)', color='red')
    ax_temp.set_ylim(0, 500)
    ax_temp.plot(r_centers, np.mean(temp_eV, axis=0), color='red', linestyle='-', linewidth=2,
                 label='Plasma Temperature')
    ax_temp.tick_params(axis='y', labelcolor='red')

    lines_1, labels_1 = ax_1d_avg.get_legend_handles_labels()
    lines_2, labels_2 = ax_temp.get_legend_handles_labels()
    ax_1d_avg.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    ax_1d_a1.legend(loc='upper right')

    # === SAVE 1: CLEAN VIDEO FRAME ===
    fig.savefig(out_img_clean, dpi=100, bbox_inches='tight')

    plt.close(fig)


# --- 5. EXECUTE MULTIPROCESSING ---
if __name__ == '__main__':
    total_frames = len(vtk_files)
    num_cores = max(1, cpu_count() - 2)

    print(">>> Pre-calculating 90° slice history for time-series graph (this may take a minute)...")
    time_hist_list = []
    rho_hist_list = []

    # Read through all files once to build the time-series arrays
    for i, filename in enumerate(vtk_files):
        filepath = os.path.join(data_dir, filename)
        _, _, _, data = athena_read.vtk(filepath)

        # Calculate the average density at the user-defined 90-degree slice for this frame
        avg_rho = np.mean(data['rho'][0, idx_phi_1, :])

        time_hist_list.append(get_sim_time(filepath, i))
        rho_hist_list.append(avg_rho)

    global_time_hist = np.array(time_hist_list)
    global_rho_hist = np.array(rho_hist_list)

    print(f"Starting parallel render across {num_cores} cores for {total_frames} frames...")
    render_start_time = time.time()

    # Pack the historical arrays into the arguments for the parallel workers
    tasks = [(i, filename, global_time_hist, global_rho_hist) for i, filename in enumerate(vtk_files)]

    with Pool(processes=num_cores) as pool:
        for i, _ in enumerate(pool.imap_unordered(render_frame, tasks), 1):
            elapsed = time.time() - render_start_time
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total_frames - i) / rate if rate > 0 else 0
            bar_len = 30
            filled_len = int(bar_len * i / total_frames)
            bar = '█' * filled_len + '-' * (bar_len - filled_len)
            sys.stdout.write(
                f"\rRendering: [{bar}] {i}/{total_frames} | {rate:.1f} fps | ETA: {int(eta // 60):02d}:{int(eta % 60):02d}   ")
            sys.stdout.flush()

    print("\n>>> Stitching with FFmpeg...")

    ffmpeg_base = [
        'ffmpeg', '-y', '-framerate', '24',
        '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
        '-profile:v', 'high', '-crf', '18'
    ]

    try:
        # Stitch Clean Video
        cmd_clean = ffmpeg_base[:4] + ['-i', 'temp_frames/clean_%05d.png'] + ffmpeg_base[4:] + [
            '[Video] torus_clean.mp4']
        subprocess.run(cmd_clean, cwd=data_dir, check=True)
        print(">>> Success! Clean video saved as [Video] torus_clean.mp4")

    except subprocess.CalledProcessError as e:
        print(f"❌ FFmpeg failed with error code: {e.returncode}")