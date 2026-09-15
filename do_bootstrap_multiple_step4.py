import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import json
import argparse
from collections import defaultdict
import mplhep as hep
hep.style.use("CMS")

COLORS = ['#3f90da', '#ffa90e', '#bd1f01', '#94a4a2', '#832db6',
          '#a96b59', '#e76300', '#b9ac70', '#717581', '#92dadd']


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Combine step3 runs and plot resolution vs BV')
    parser.add_argument('--input-dirs', nargs='+', required=True,
                        help='List of step3 output directories, each containing resolution_summary.json')
    parser.add_argument('--output-dir', type=str, default='step4_output',
                        help='Output directory for plots')
    parser.add_argument('--by-etroc', action =  'store_true',
                        help='Output directory for plots')
    parser.add_argument('--isolate-etrocs', nargs = '*', default = [],
                        dest = 'isolated_etroc_nos',
                        help='Output directory for plots')
    parser.add_argument('--isolate-module', nargs = '*', default = [],
                        help='Output directory for plots')
    args = parser.parse_args()
    args.isolate_etrocs = [f'etroc{i}' for i in args.isolated_etroc_nos]
    os.makedirs(args.output_dir, exist_ok=True)

    # Collect per-module data across all runs
    # Keys: module name → lists of (bv, mean, std, temperature)
    module_data = defaultdict(lambda: {'bv': [], 'mean': [], 'std': [], 'temperature': []})

    for d in args.input_dirs:
        if args.by_etroc:
            possetrocs = [f'etroc{i}' for i in range(4)]
            etrocs = [f for f in os.listdir(d) if f in possetrocs]
            if len(etrocs) > 0:
                print('Found these etrocs:', etrocs)
                for e in etrocs:
                    if len(args.isolate_etrocs) > 0 and not e in args.isolate_etrocs:
                        print(e, args.isolate_etrocs, e in args.isolate_etrocs)
                        continue 
                    summary_file = os.path.join(d, e, 'resolution_summary.json')
                    if not os.path.exists(summary_file):
                        print(f"WARNING: {summary_file} not found — skipping")
                        continue
                    with open(summary_file) as f:
                        summary = json.load(f)
                    temperature = summary.get('temperature', None)
                    for layer in ['i', 'j', 'k']:
                        mod = summary['modules'][layer]
                        if mod['mean_sigma_ps'] is None:
                            continue
                        name = mod['name']
                        if len(args.isolate_module) > 0 and not name in args.isolate_module:
                            continue
                        module_data[name + '_' + e]['bv'].append(mod['bias_voltage'])
                        module_data[name + '_' + e]['mean'].append(mod['mean_sigma_ps'])
                        module_data[name + '_' + e]['std'].append(mod['std_sigma_ps'])
                        module_data[name + '_' + e]['temperature'].append(temperature)
        else:
            summary_file = os.path.join(d, 'resolution_summary.json')
            if not os.path.exists(summary_file):
                print(f"WARNING: {summary_file} not found — skipping")
                continue
            with open(summary_file) as f:
                summary = json.load(f)

            temperature = summary.get('temperature', None)

            for layer in ['i', 'j', 'k']:
                mod = summary['modules'][layer]
                if mod['mean_sigma_ps'] is None:
                    continue
                name = mod['name']
                module_data[name]['bv'].append(mod['bias_voltage'])
                module_data[name]['mean'].append(mod['mean_sigma_ps'])
                module_data[name]['std'].append(mod['std_sigma_ps'])
                module_data[name]['temperature'].append(temperature)

    if not module_data:
        print("No valid data found. Exiting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Plot: resolution vs bias voltage
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for idx, module_name in enumerate(sorted(module_data.keys())):
        data    = module_data[module_name]
        bv      = np.array(data['bv'])
        mean    = np.array(data['mean'])
        std     = np.array(data['std'])
        sort_i  = np.argsort(bv)
        bv, mean, std = bv[sort_i], mean[sort_i], std[sort_i]

        ax.errorbar(bv, mean, yerr=std,
                    fmt='o-', color=COLORS[idx % len(COLORS)],
                    capsize=4, linewidth=2, markersize=8,
                    label=module_name.replace('_etroc', '- ETROC '))

    ax.set_xlabel('Bias Voltage (V)')
    ax.set_ylabel(r'$\langle\sigma\rangle$ (ps)')
    ax.legend()
    hep.cms.text(exp="SPS July TB", text="", ax=ax)

    fig.savefig(os.path.join(args.output_dir, 'resolution_vs_BV.pdf'))
    fig.savefig(os.path.join(args.output_dir, 'resolution_vs_BV.png'), dpi=150)
    plt.close(fig)
    print(f"Saved: {args.output_dir}/resolution_vs_BV.png")

    # -------------------------------------------------------------------------
    # Plot: resolution vs temperature (if available)
    # -------------------------------------------------------------------------

    has_temperature = any(
        t is not None
        for data in module_data.values()
        for t in data['temperature']
    )

    if has_temperature:
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        for idx, module_name in enumerate(sorted(module_data.keys())):
            data  = module_data[module_name]
            temps = np.array([t for t in data['temperature'] if t is not None], dtype=float)
            means = np.array([m for t, m in zip(data['temperature'], data['mean']) if t is not None])
            stds  = np.array([s for t, s in zip(data['temperature'], data['std'])  if t is not None])
            sort_i = np.argsort(temps)
            temps, means, stds = temps[sort_i], means[sort_i], stds[sort_i]

            ax.errorbar(temps, means, yerr=stds,
                        fmt='o-', color=COLORS[idx % len(COLORS)],
                        capsize=4, linewidth=2, markersize=8,
                        label=module_name.replace('_etroc', '- ETROC '))

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel(r'$\langle\sigma\rangle$ (ps)')
        ax.legend()
        hep.cms.text(exp="SPS July TB", text="", ax=ax)

        fig.savefig(os.path.join(args.output_dir, 'resolution_vs_temperature.pdf'))
        fig.savefig(os.path.join(args.output_dir, 'resolution_vs_temperature.png'), dpi=150)
        plt.close(fig)
        print(f"Saved: {args.output_dir}/resolution_vs_temperature.png")
