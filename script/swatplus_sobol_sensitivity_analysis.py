import numpy as np
import shutil
import subprocess
import logging
from pathlib import Path
import pandas as pd
import time
from SALib.sample import sobol
from SALib.analyze import sobol as sobol_analyze
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# CONFIGURATION
# ============================================================
# Paths
base_dir = Path(r"path\to\your\TxtInOut")          # Original SWAT+ model folder
swat_executable = Path(r"path\to\swat_exe")               # SWAT+ executable

output_dir = Path(r"path\to\output_folder")          # Where results will be saved
target_channel = "cha038"                                # Channel name to extract flow from
log_file = Path(r"path\to\log_file.txt")

# Analysis settings
N           = 128     # Base sample size
MAX_WORKERS = 6       # Number of parallel processes
SAVE_EVERY  = 100     # Save intermediate results every N completed runs

# ============================================================
# PROBLEM DEFINITION
# ============================================================
problem = {
    'num_vars': 8,
    'names': [
        "cn2", "esco", "epco", "canmx",
        "awc", "k", "perco", "surlag"
    ],
    'bounds': [
        [35.0, 95.0],    # cn2
        [0.01, 1.0],     # esco
        [0.01, 1.0],     # epco
        [0.0, 100.0],    # canmx
        [0.05, 0.40],    # awc
        [1.0, 150.0],    # k
        [0.0, 1.0],      # perco
        [0.05, 24.0],    # surlag
    ]
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def create_run_dir(run_dir: Path) -> bool:
    try:
        if run_dir.exists():
            shutil.rmtree(run_dir)
        shutil.copytree(base_dir, run_dir)
        return True
    except Exception as e:
        print(f"create_run_dir failed: {e}")
        return False


def write_calibration_cal(run_dir: Path, params: dict) -> bool:
    try:
        lines = []
        # Header line (keep consistent with SWAT+ editor style)
        lines.append("calibration.cal: written by SWAT+ editor v2.1.3\n")
        lines.append(f"{len(params)}\n")
        lines.append("cal_parm               chg_typ       chg_val     conds  soil_lyr1  soil_lyr2       yr1       yr2      day1      day2   obj_tot\n")

        for name, value in params.items():
            # Decide change type
            if name in ["awc", "k", "revap_min", "latq_co"]:
                chg_typ = "pctchg"
            else:
                chg_typ = "absval"

            # Format line with 11 fields
            line = f"{name:<22}{chg_typ:<12}{value:10.5f}     0         0         0         0         0         0         0         0\n"
            lines.append(line)

        cal_file = run_dir / "calibration.cal"
        with open(cal_file, "w") as f:
            f.writelines(lines)

        return True
    except Exception as e:
        print(f"write_calibration_cal failed: {e}")
        return False



def run_swat(run_dir: Path) -> bool:
    try:
        subprocess.run(
            [str(swat_executable)],
            cwd=str(run_dir),
            check=True,
            capture_output=True,
            text=True,
            timeout=900
        )
        return True
    except Exception as e:
        print(f"SWAT failed in {run_dir.name}: {e}")
        return False


def extract_flow(run_dir: Path):
    output_file = run_dir / "channel_sd_mon.txt"
    if not output_file.exists():
        return None
    try:
        df = pd.read_csv(
            output_file,
            sep=r"\s+",
            skiprows=[0, 2],
            header=0,
            engine="python"
        )
        df.columns = df.columns.str.strip().str.lower()
        if "name" not in df.columns or "flo_out" not in df.columns:
            return None
        channel_data = df[df["name"].str.strip().str.lower() == target_channel.lower()]
        if channel_data.empty:
            return None
        return float(channel_data["flo_out"].mean())
    except Exception as e:
        print(f"Error extracting flow: {e}")
        return None


def process_one_run(args):
    i, params = args
    run_dir = output_dir / f"run_{i:04d}"

    avg_flow = np.nan
    success = False

    try:
        if create_run_dir(run_dir):
            param_dict = dict(zip(problem['names'], params))
            if write_calibration_cal(run_dir, param_dict):
                if run_swat(run_dir):
                    avg_flow = extract_flow(run_dir)
                    if avg_flow is not None:
                        success = True
    except Exception as e:
        print(f"Run {i} exception: {e}")

    # Delete run folder to save space
    if run_dir.exists():
        shutil.rmtree(run_dir, ignore_errors=True)

    return {
        "run": i + 1,
        "params": params.tolist(),
        "flow": avg_flow if success else np.nan,
        "success": success
    }


def save_results(results_list, filename):
    param_names = problem['names']
    clean_rows = []
    for r in results_list:
        row = {"run": r["run"]}
        for name, value in zip(param_names, r["params"]):
            row[name] = value
        row["flow"] = r["flow"]
        row["success"] = r["success"]
        clean_rows.append(row)
    df = pd.DataFrame(clean_rows)
    df.to_csv(filename, index=False)
    return df


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    start_time = time.time()   # record start

    if log_file.exists():
        try:
            log_file.unlink()
        except PermissionError:
            pass

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    log = logging.getLogger()

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Sampling
    np.random.seed(42)
    param_values = sobol.sample(problem, N, calc_second_order=True)
    n_runs = param_values.shape[0]
    log.info(f"Generated {n_runs} parameter sets (N={N})")
    log.info(f"Using {MAX_WORKERS} parallel workers")

    # Parallel loop
    results = []
    completed = 0
    log.info(f"Starting {n_runs} runs...")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one_run, (i, params)): i
            for i, params in enumerate(param_values)
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1

            status = "OK" if result["success"] else "FAILED"
            flow_str = f"{result['flow']:.4f}" if result["success"] else "NaN"
            print(f"Run {result['run']:4d}/{n_runs} | flow = {flow_str} | {status}")

            if completed % SAVE_EVERY == 0 or completed == n_runs:
                temp_sorted = sorted(results, key=lambda x: x["run"])
                partial_file = output_dir / f"sensitivity_results_partial_{completed:04d}.csv"
                save_results(temp_sorted, partial_file)
                log.info(f"Intermediate results saved -> {partial_file.name}")

    results.sort(key=lambda x: x["run"])

    # Final save
    final_csv = output_dir / "sensitivity_results.csv"
    save_results(results, final_csv)
    log.info(f"Final results saved -> {final_csv}")

    # Diagnostics
    outputs = np.array([r["flow"] for r in results], dtype=float)
    n_ok = np.isfinite(outputs).sum()
    n_nan = np.isnan(outputs).sum()

    print("\n" + "="*60)
    print("FINAL DIAGNOSTICS")
    print("="*60)
    print(f"Total runs       : {len(outputs)}")
    print(f"Successful runs  : {n_ok}")
    print(f"Failed / NaN     : {n_nan}")
    if n_ok > 0:
        print(f"Min / Max / Mean : {np.nanmin(outputs):.4f} / {np.nanmax(outputs):.4f} / {np.nanmean(outputs):.4f}")
        print(f"Std              : {np.nanstd(outputs):.6f}")

        # Sobol analysis
    valid_mask = np.isfinite(outputs)
    n_valid = valid_mask.sum()
    print(f"\nValid runs available for Sobol: {n_valid} / {len(outputs)}")

    if n_valid >= 50:
        outputs_clean = outputs.copy()
        outputs_clean[~valid_mask] = np.nanmean(outputs)

        Si = sobol_analyze.analyze(problem, outputs_clean, calc_second_order=True, print_to_console=False)

        print("\nFirst-order indices (S1):")
        for name, s1 in zip(problem['names'], Si['S1']):
            print(f"  {name:15s}: {s1:8.4f}")

        print("\nTotal-order indices (ST):")
        for name, st in zip(problem['names'], Si['ST']):
            print(f"  {name:15s}: {st:8.4f}")

        pd.DataFrame({
            "parameter": problem['names'],
            "S1": Si['S1'],
            "ST": Si['ST']
        }).to_csv(output_dir / "sobol_indices.csv", index=False)

        print("\nSobol indices saved to sobol_indices.csv")

        # ============================================================
        # DASHBOARD
        # ============================================================
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"SWAT+ Sensitivity Analysis – Final Results (N={N})", 
                     fontsize=18, color='cyan', fontweight='bold')

        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)

        # Panel 1: Summary
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.axis('off')
        summary = (
            f"Total runs      : {len(outputs)}\n"
            f"Successful      : {n_ok}\n"
            f"Failed          : {n_nan}\n\n"
            f"Min flow        : {np.nanmin(outputs):.3f}\n"
            f"Max flow        : {np.nanmax(outputs):.3f}\n"
            f"Mean flow       : {np.nanmean(outputs):.3f}\n"
            f"Std             : {np.nanstd(outputs):.3f}"
        )
        ax1.text(0.05, 0.95, summary, transform=ax1.transAxes, fontsize=13,
                 verticalalignment='top', fontfamily='monospace', color='white')

        # Panel 2: Histogram
        ax2 = fig.add_subplot(gs[0, 1])
        valid = outputs[np.isfinite(outputs)]
        ax2.hist(valid, bins=30, color='#ff6b6b', edgecolor='white', alpha=0.85)
        ax2.axvline(np.mean(valid), color='yellow', linestyle='--', linewidth=2)
        ax2.set_title("Flow Distribution", color='white')
        ax2.set_xlabel("flo_out")

        # Panel 3: S1
        ax3 = fig.add_subplot(gs[0, 2])
        s1 = np.nan_to_num(Si['S1'])
        s1_plot = np.clip(s1, 0, 1.5)
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(s1)))
        bars = ax3.barh(problem['names'], s1_plot, color=colors, edgecolor='white')
        ax3.set_xlim(0, max(0.6, s1_plot.max() * 1.15))
        ax3.set_title("First-order (S1)", color='white')
        for bar, val in zip(bars, s1):
            ax3.text(min(val, 1.4) + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=9, color='white')

        # Panel 4: ST
        ax4 = fig.add_subplot(gs[1, 2])
        st = np.nan_to_num(Si['ST'])
        st_plot = np.clip(st, 0, 1.5)
        colors = plt.cm.magma(np.linspace(0.2, 0.9, len(st)))
        bars = ax4.barh(problem['names'], st_plot, color=colors, edgecolor='white')
        ax4.set_xlim(0, max(0.6, st_plot.max() * 1.15))
        ax4.set_title("Total-order (ST)", color='white')
        for bar, val in zip(bars, st):
            ax4.text(min(val, 1.4) + 0.01, bar.get_y() + bar.get_height()/2,
                     f"{val:.3f}", va='center', fontsize=9, color='white')

        # Panel 5: Scatter – ALL parameters
        ax5 = fig.add_subplot(gs[1, :2])
        p = np.array([r['params'] for r in results])
        f = outputs
        mask = np.isfinite(f)

        cmaps = ["plasma", "viridis", "cividis", "magma", "inferno", "cool", "spring", "winter"]
        markers = ["o", "X", "s", "D", "^", "v", "P", "*"]

        for i, name in enumerate(problem['names']):
            ax5.scatter(
                p[mask, i], f[mask],
                c=f[mask], cmap=cmaps[i % len(cmaps)],
                s=30, alpha=0.65,
                marker=markers[i % len(markers)],
                label=name, edgecolors="white", linewidth=0.25
            )

        ax5.set_xlabel("Parameter value")
        ax5.set_ylabel("flo_out")
        ax5.legend(loc="upper right", fontsize=9, framealpha=0.7)
        ax5.set_title("All parameters vs Flow", color="white")

        # Save dashboard
        plt.subplots_adjust(left=0.08, right=0.95, top=0.90, bottom=0.08, hspace=0.35, wspace=0.30)
        dashboard_path = output_dir / "sensitivity_dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"\nDashboard saved to: {dashboard_path}")
        plt.close(fig)
        # ============================================================
        # END OF DASHBOARD
        # ============================================================

    else:
        print("Too few valid runs for Sobol analysis.")

    # Total time
    end_time = time.time()
    elapsed = end_time - start_time
    print("\n" + "="*60)
    print(f"TOTAL EXECUTION TIME: {elapsed:.2f} seconds")
    print("="*60)

    print("\nDone!")
