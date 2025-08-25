import os
import shutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.colors import to_rgba
from pathlib import Path
from PIL import Image

# ============ CONFIG ============ #
BASE_DIR = Path(__file__).parent
STELLAR_IMG_DIR = BASE_DIR / "Images"
DATA_PATH = BASE_DIR / "BSE_Detailed_Output_0.h5"
CSV_PATH = BASE_DIR / "detailed_compas.csv"
CE_IMG_PATH = BASE_DIR / "Images" / "common_envelope.png"
BG_IMG_PATH = BASE_DIR / "Images" / "Background.png"

TARGET_FRAMES_PER_PHASE = 25
LARGE_JUMP_THRESHOLD = 0.2
EXTRA_INTERP_FRAMES = 10

# ============ HELPER FUNCTIONS ============ #

def apply_min_radius(radii, min_radius=0.5):
    """Floor radii to a minimum value."""
    return np.maximum(radii, min_radius)

def csv_stellarTypeName(idx):
    stellarTypes = ['MS', 'MS', 'HG', 'FGB', 'CHeB', 'EAGB', 'TPAGB', 'HeMS', 'HeHG',
                    'HeGB', 'HeWD', 'COWD', 'ONeWD', 'NS', 'BH', 'MR', 'CHE']
    return stellarTypes[idx] if idx < len(stellarTypes) else 'unknown'

def interpolate_numeric_rows(row_start, row_end, numeric_cols, n_interp):
    """Interpolate numeric columns linearly between two rows."""
    interpolated = []
    for step in range(1, n_interp + 1):
        alpha = step / (n_interp + 1)
        interp_row = {}
        for col in numeric_cols:
            interp_row[col] = (1 - alpha) * row_start[col] + alpha * row_end[col]
        interpolated.append(interp_row)
    return interpolated

def interpolate_phase_rows(phase_df):
    """Interpolate or sample rows in a phase to reach TARGET_FRAMES_PER_PHASE."""
    n_rows = len(phase_df)
    numeric_cols = ['time', 'a', 'e', 'm1', 'm2', 'radius1', 'radius2']
    non_numeric_cols = ['stype1', 'stype2', 'eventString', 'dm1', 'dm2']
    
    if n_rows == 1:
        # Repeat the single row to fill TARGET_FRAMES_PER_PHASE frames
        repeated = [phase_df.iloc[0].to_dict()] * TARGET_FRAMES_PER_PHASE
        return pd.DataFrame(repeated)

    elif n_rows == TARGET_FRAMES_PER_PHASE:
        return phase_df.copy()

    elif n_rows > TARGET_FRAMES_PER_PHASE:
        indices = np.linspace(0, n_rows - 1, TARGET_FRAMES_PER_PHASE).astype(int)
        sampled = phase_df.iloc[indices].copy()
        sampled['stype1'] = sampled['stype1'].astype(int)
        sampled['stype2'] = sampled['stype2'].astype(int)
        return sampled.reset_index(drop=True)

    else:
        # Interpolate between rows to reach desired length
        rows_out = []
        for i in range(n_rows - 1):
            row_start = phase_df.iloc[i]
            row_end = phase_df.iloc[i + 1]
            rows_out.append(row_start.to_dict())

            n_interp_exact = (TARGET_FRAMES_PER_PHASE - 1) / (n_rows - 1) - 1
            n_interp = max(int(round(n_interp_exact)), 0)
            interp_rows = interpolate_numeric_rows(row_start, row_end, numeric_cols, n_interp)

            for interp_row in interp_rows:
                for col in non_numeric_cols:
                    interp_row[col] = row_start[col]
                interp_row['stype1'] = int(interp_row['stype1'])
                interp_row['stype2'] = int(interp_row['stype2'])
                rows_out.append(interp_row)

        rows_out.append(phase_df.iloc[-1].to_dict())
        return pd.DataFrame(rows_out)

'''
    if n_rows == TARGET_FRAMES_PER_PHASE:
        return phase_df.copy()
    elif n_rows > TARGET_FRAMES_PER_PHASE:
        indices = np.linspace(0, n_rows - 1, TARGET_FRAMES_PER_PHASE).astype(int)
        sampled = phase_df.iloc[indices].copy()
        sampled['stype1'] = sampled['stype1'].astype(int)
        sampled['stype2'] = sampled['stype2'].astype(int)
        return sampled.reset_index(drop=True)
    else:
        rows_out = []
        for i in range(n_rows - 1):
            row_start = phase_df.iloc[i]
            row_end = phase_df.iloc[i + 1]
            rows_out.append(row_start.to_dict())
            # Calculate number of interpolation steps between these rows
            n_interp_exact = (TARGET_FRAMES_PER_PHASE - 1) / (n_rows - 1) - 1
            n_interp = max(int(round(n_interp_exact)), 0)
            interp_rows = interpolate_numeric_rows(row_start, row_end, numeric_cols, n_interp)
            for interp_row in interp_rows:
                # Copy non-numeric columns from start row
                for col in non_numeric_cols:
                    interp_row[col] = row_start[col]
                # Ensure integer types for stellar types
                interp_row['stype1'] = int(interp_row['stype1'])
                interp_row['stype2'] = int(interp_row['stype2'])
                rows_out.append(interp_row)
        rows_out.append(phase_df.iloc[-1].to_dict())
        return pd.DataFrame(rows_out)
'''



def insert_extra_interpolations(df, threshold=LARGE_JUMP_THRESHOLD, extra_frames=EXTRA_INTERP_FRAMES):
    """
    Detect large jumps between frames and insert extra interpolation frames to smooth transitions.
    """
    numeric_cols = ['time', 'a', 'e', 'm1', 'm2', 'radius1', 'radius2']
    non_numeric_cols = ['stype1', 'stype2', 'eventString', 'dm1', 'dm2']
    rows_out = []
    for i in range(len(df) - 1):
        row_start = df.iloc[i]
        row_end = df.iloc[i + 1]
        rows_out.append(row_start.to_dict())
        jump_detected = any(
            abs(row_end[col] - row_start[col]) / max(abs(row_start[col]), 1e-8) >= threshold
            for col in ['a', 'radius1', 'radius2']
        )
        if jump_detected:
            interp_rows = interpolate_numeric_rows(row_start, row_end, numeric_cols, extra_frames)
            for interp_row in interp_rows:
                for col in non_numeric_cols:
                    interp_row[col] = row_start[col]
                interp_row['stype1'] = int(interp_row['stype1'])
                interp_row['stype2'] = int(interp_row['stype2'])
                rows_out.append(interp_row)
    rows_out.append(df.iloc[-1].to_dict())
    return pd.DataFrame(rows_out)

def set_font_params(use_latex=True):
    use_latex = use_latex and (shutil.which("latex") is not None)
    rcParams.update({
        "font.serif": "Times New Roman",
        "text.usetex": use_latex,
        "axes.grid": True,
        "grid.color": "gray",
        "grid.linestyle": ":",
        "axes.titlesize": 18,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.framealpha": 1,
    })

# ============ CSV Data Loading & Event Classes ============ #

class CSVEvent:
    def __init__(self, row):
        self.time = row['time']
        self.a = row['a']
        self.e = row['e']
        self.m1 = row['m1']
        self.m2 = row['m2']
        self.radius1 = row['radius1']
        self.radius2 = row['radius2']
        self.stype1 = int(row['stype1'])
        self.stype2 = int(row['stype2'])
        self.stypeName1 = csv_stellarTypeName(self.stype1)
        self.stypeName2 = csv_stellarTypeName(self.stype2)
        self.dm1 = row.get('dm1', 0.0)
        self.dm2 = row.get('dm2', 0.0)
        self.eventString = row.get('eventString', f"{self.stypeName1} + {self.stypeName2}")

def load_csv_events(csv_path=None):
    import h5py

    with h5py.File(DATA_PATH, 'r') as f:
        # Use only binary evolution records
        mask = f['Record_Type'][()] == 4
        data = {key: f[key][()][mask] for key in f.keys()}

    # Apply minimum radius floor
    radius1 = apply_min_radius(data['Radius(1)'])
    radius2 = apply_min_radius(data['Radius(2)'])

    stype1 = data['Stellar_Type(1)'].astype(int)
    stype2 = data['Stellar_Type(2)'].astype(int)

    # Build dataframe
    df = pd.DataFrame({
        'time': data['Time'],
        'a': data['SemiMajorAxis'],
        'e': data['Eccentricity'],
        'radius1': radius1,
        'radius2': radius2,
        'm1': data['Mass(1)'],
        'm2': data['Mass(2)'],
        'stype1': stype1,
        'stype2': stype2,
        'eventString': [f"{csv_stellarTypeName(s1)} + {csv_stellarTypeName(s2)}" for s1, s2 in zip(stype1, stype2)],
        'dm1': data['dM1/dt'] if 'dM1/dt' in data else np.zeros_like(data['Time']),
        'dm2': data['dM2/dt'] if 'dM2/dt' in data else np.zeros_like(data['Time']),
    })

    # Group by phases where stellar types change
    df['stype_pair'] = list(zip(df['stype1'], df['stype2']))
    phase_groups = (df['stype_pair'] != df['stype_pair'].shift()).cumsum()

    # Interpolate phases
    phases = [interpolate_phase_rows(phase_df) for _, phase_df in df.groupby(phase_groups)]
    df_interpolated = pd.concat(phases, ignore_index=True)

    # Insert extra interpolations at large jumps
    df_final = insert_extra_interpolations(df_interpolated)

    # Drop any rows with NaNs in stellar types
    df_final = df_final.dropna(subset=['stype1', 'stype2'])
    df_final['stype1'] = df_final['stype1'].astype(int)
    df_final['stype2'] = df_final['stype2'].astype(int)

    # Save for inspection
    output_csv_path = BASE_DIR / "compas_processed.csv"
    df_final.to_csv(output_csv_path, index=False)
    print(f"Saved interpolated CSV data to {output_csv_path}")

    # Convert to CSVEvent objects
    return [CSVEvent(row) for _, row in df_final.iterrows()]

# ============ Animation & Visualization ============ #

# Load CE overlay image once globally if exists
ce_overlay_img = None
if CE_IMG_PATH.exists():
    ce_overlay_img = Image.open(CE_IMG_PATH).convert("RGBA")
else:
    print(f"Warning: CE overlay image not found at {CE_IMG_PATH}")

def log_scaled_radius(radius_rsun, pixels_at_ref=100):
    """
    Logarithmically scale radius to pixel size.
    """
    radius_rsun = np.maximum(radius_rsun, 1e-3)
    log_scale = np.log10(radius_rsun + 1)
    reference_log = np.log10(100 + 1)
    return (log_scale / reference_log) * pixels_at_ref

def get_orbit_pos(a, e, t_frac):
    """
    Compute star position in orbit at fraction t_frac of orbital period.
    """
    b = a * np.sqrt(1 - e ** 2)
    theta = 2 * np.pi * t_frac
    x_rsun = a * np.cos(theta) - a * e
    y_rsun = b * np.sin(theta)
    dist_rsun = np.sqrt(x_rsun**2 + y_rsun**2)
    scale = log_scaled_radius(dist_rsun) / dist_rsun if dist_rsun > 0 else 0
    return x_rsun * scale, y_rsun * scale

def get_extent(center_x, center_y, img_array):
    """
    Get extent box for imshow based on center position and image size.
    """
    h, w = img_array.shape[:2]
    axis_scale = h / 2
    return [center_x - axis_scale * (w / h), center_x + axis_scale * (w / h),
            center_y - axis_scale, center_y + axis_scale]

def scale_image(img, radius_rsun):
    """
    Scale star image to radius in pixels using log scale.
    """
    radius_px = max(log_scaled_radius(radius_rsun), 5)
    w, h = img.size
    scale_factor = (2 * radius_px) / h
    new_w = max(int(w * scale_factor), 5)
    new_h = max(int(h * scale_factor), 5)
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def cubic_bezier(p0, p1, p2, p3, n_points=100):
    """
    Compute points along cubic Bézier curve.
    """
    t = np.linspace(0, 1, n_points)
    curve = ((1 - t)**3) * p0[:, None] + \
            3 * ((1 - t)**2) * t * p1[:, None] + \
            3 * (1 - t) * (t**2) * p2[:, None] + \
            (t**3) * p3[:, None]
    return curve

def draw_mass_transfer_stream_hourglass(ax, start_pos, end_pos, radius_start, radius_end):
    """
    Draw a mass transfer stream shaped like an hourglass between stars.
    """
    x1, y1 = start_pos
    x4, y4 = end_pos

    vec = np.array([x4 - x1, y4 - y1])
    length = np.linalg.norm(vec)
    if length == 0:
        return None

    unit_vec = vec / length
    perp_vec = np.array([-unit_vec[1], unit_vec[0]])

    width_start = 0.8 * (2 * radius_start)
    width_end = 0.8 * (2 * radius_end)

    p1 = np.array([x1, y1]) + (width_start / 2) * perp_vec
    p2 = np.array([x1, y1]) - (width_start / 2) * perp_vec
    p3 = np.array([x4, y4]) - (width_end / 2) * perp_vec
    p4 = np.array([x4, y4]) + (width_end / 2) * perp_vec

    neck_frac = 0.5
    neck_width_frac = 0.4
    neck_center = np.array([x1, y1]) + vec * neck_frac
    neck_width_start = width_start * neck_width_frac
    neck_width_end = width_end * neck_width_frac
    neck_width = (neck_width_start + neck_width_end) / 2

    p_mid_upper = neck_center + (neck_width / 2) * perp_vec
    p_mid_lower = neck_center - (neck_width / 2) * perp_vec

    left_curve = cubic_bezier(
        p1,
        p1 + (vec * 0.35) - (perp_vec * width_start * 0.25),
        p_mid_upper + (perp_vec * neck_width * 0.15),
        p_mid_upper
    )

    right_curve = cubic_bezier(
        p2,
        p2 + (vec * 0.35) + (perp_vec * width_start * 0.25),
        p_mid_lower - (perp_vec * neck_width * 0.15),
        p_mid_lower
    )

    end_upper_line = np.linspace(p_mid_upper, p4, left_curve.shape[1])
    end_lower_line = np.linspace(p_mid_lower, p3, left_curve.shape[1])[::-1]

    polygon_points = np.vstack([
        left_curve.T,
        end_upper_line,
        end_lower_line,
        right_curve.T[::-1]
    ])

    base_color = np.array(to_rgba('#fffbcf'))
    patch = Polygon(polygon_points, closed=True, color=base_color, alpha=0.4, zorder=2)
    patch.is_overlay = True
    ax.add_patch(patch)
    return patch

def animate_evolution(events, outdir='.', use_latex=False, bg_img_path=None, save_mode=True):
    """
    Animate the binary star evolution given list of CSVEvent objects.
    """
    dpi = 100
    fig_width_in, fig_height_in = 20, 20
    fig, ax = plt.subplots(figsize=(fig_width_in, fig_height_in), dpi=dpi)
    ax.set_facecolor('black')
    ax.axis('off')

    # Calculate plotting boundaries based on max radius and separation
    max_radius = max(max(e.radius1, e.radius2) for e in events)
    max_sep = max(e.a * (1 + e.e) for e in events if (e.m1 + e.m2) > 0)
    max_extent = max_radius * 4 + max_sep
    full_extent_px = log_scaled_radius(max_extent)
    margin_px = 50
    limit = full_extent_px + margin_px

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal')

    # Load and display background image if provided
    if bg_img_path and Path(bg_img_path).exists():
        bg_img = Image.open(bg_img_path).convert("RGBA")
        bg_img = bg_img.resize((int(fig_width_in * dpi), int(fig_height_in * dpi)), Image.Resampling.LANCZOS)
        ax.imshow(bg_img, extent=[-limit, limit, -limit, limit], zorder=0)

    # Prepare image containers for star images
    im1 = ax.imshow(np.zeros((1, 1, 4)), extent=[0, 0, 0, 0], zorder=3)
    im2 = ax.imshow(np.zeros((1, 1, 4)), extent=[0, 0, 0, 0], zorder=3)

    text = ax.text(0, -limit * 0.9, '', color='white', fontsize=14, ha='center')

    frames = [e.__dict__ for e in events]
    frames_per_revolution = 25  # orbital frames per full revolution

    def update(frame_idx):
        f = frames[frame_idx]
        # Limit large semimajor axes for display
        f['a'] = min(f['a'], 5000)
        t_frac_orbit = (frame_idx % frames_per_revolution) / frames_per_revolution

        # Load star images
        try:
            img1 = Image.open(STELLAR_IMG_DIR / f"{f['stypeName1']}.png").convert("RGBA")
            img2 = Image.open(STELLAR_IMG_DIR / f"{f['stypeName2']}.png").convert("RGBA")
        except FileNotFoundError:
            # If image missing, skip drawing this frame
            return

        img1 = scale_image(img1, f['radius1'])
        img2 = scale_image(img2, f['radius2'])

        # Compute orbit positions
        x1, y1 = get_orbit_pos(f['a'], f['e'], t_frac_orbit)
        mass_ratio = f['m1'] / f['m2'] if f['m2'] > 0 else 1.0
        x2, y2 = -x1 * mass_ratio, -y1 * mass_ratio

        # Update images positions and extents
        im1.set_data(np.array(img1))
        im2.set_data(np.array(img2))
        im1.set_extent(get_extent(x1, y1, np.array(img1)))
        im2.set_extent(get_extent(x2, y2, np.array(img2)))

        # Draw common envelope overlay if both stars in CE phase (type 15)
        if f['stype1'] == 15 and f['stype2'] == 15 and ce_overlay_img:
            ce_scaled = scale_image(ce_overlay_img, f['a'] / 2)
            ax.imshow(np.array(ce_scaled), extent=get_extent(0, 0, np.array(ce_scaled)),
                      alpha=0.3, zorder=1)

        # Clear existing mass transfer overlays
        [p.remove() for p in ax.patches if getattr(p, 'is_overlay', False)]

        # Draw mass transfer stream if mass is flowing
        if f['dm1'] < 0 and f['dm2'] > 0:
            # Mass transfer from star 1 to 2
            draw_mass_transfer_stream_hourglass(ax, (x1, y1), (x2, y2), f['radius1'], f['radius2'])
        elif f['dm2'] < 0 and f['dm1'] > 0:
            # Mass transfer from star 2 to 1
            draw_mass_transfer_stream_hourglass(ax, (x2, y2), (x1, y1), f['radius2'], f['radius1'])

        # Update annotation text with time and event string
        time_text = f"Time = {f['time']:.3g} Myr\n{f['eventString']}"
        text.set_text(time_text)

    # Run animation
    anim = FuncAnimation(fig, update, frames=len(frames), interval=100)

    if save_mode:
        output_path = BASE_DIR / "binary_evolution_animation.mp4"
        print(f"Saving animation to {output_path}")
        anim.save(output_path, dpi=dpi, fps=20)
    else:
        plt.show()

# ============ MAIN ENTRY POINT ============ #

def main():
    # Optional: Set LaTeX font params
    set_font_params(use_latex=False)

    # Load events from CSV
    events = load_csv_events(CSV_PATH)

    # Animate evolution
    animate_evolution(events, outdir='.', use_latex=False, bg_img_path=BG_IMG_PATH, save_mode=True)

if __name__ == "__main__":
    main()
