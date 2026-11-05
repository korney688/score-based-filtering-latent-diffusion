"""Plot paired PSNR differences from three corpus seeds."""
from pathlib import Path
import argparse
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from build_consultant_report import validate_internal, validate_external

PROJECT = Path(__file__).resolve().parents[1]
DATASETS = {'colored_mnist': 'Colored MNIST', 'cifar10': 'CIFAR-10', 'imagenet100': 'ImageNet-100'}
SEEDS = [42, 43, 44]
STRATEGIES = {'topk': 'Top-K', 'quantile': 'Quantile-spread', 'random': 'Random'}
FRACTIONS = [.1, .2, .3]
COLORS = {'Top-K': '#292929', 'Quantile-spread': '#C87D20', 'Random': '#9C4E96'}
MARKERS = {'Top-K': 'o', 'Quantile-spread': 's', 'Random': '^'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def paired_deltas(data):
    """Pair each filtered observation with Full of the same dataset/model/seed."""
    seed_rows, summary_rows = [], []
    for (dataset, benchmark, architecture), group in data.groupby(['Dataset', 'Benchmark', 'Architecture'], sort=False):
        require(len(group) == 30 and not group.duplicated(['Seed', 'Variant']).any(), 'Expected ten variants for each of three seeds')
        require(np.isfinite(group.PSNR).all(), 'Missing/nonfinite PSNR')
        full = group[group.Variant.eq('full')].set_index('Seed')
        require(len(full) == 3 and set(full.index) == set(SEEDS), 'Full must be unique for seeds42/43/44')
        require(full.Fraction.eq(1.).all() and full['Delta PSNR vs Full'].eq(0.).all(), 'Invalid Full baseline')
        for prefix, strategy in STRATEGIES.items():
            for fraction in FRACTIONS:
                variant = f'{prefix}_p{round(fraction*100)}'
                part = group[group.Variant.eq(variant)].set_index('Seed')
                require(len(part) == 3 and set(part.index) == set(SEEDS), 'Incomplete paired point: '+variant)
                require(part.Fraction.eq(fraction).all(), 'Fraction/variant mismatch')
                values = []
                for seed in SEEDS:
                    psnr = float(part.loc[seed, 'PSNR'])
                    baseline = float(full.loc[seed, 'PSNR'])
                    delta = psnr - baseline
                    require(np.isclose(delta, part.loc[seed, 'Delta PSNR vs Full'], rtol=0, atol=1e-12), 'Stored Delta differs from same-seed Full pairing')
                    values.append(delta)
                    seed_rows.append(dict(Dataset=dataset, Benchmark=benchmark, Architecture=architecture, Strategy=strategy,
                                          Fraction=fraction, Seed=seed, PSNR=psnr, Full_PSNR_same_seed=baseline, Delta_PSNR=delta))
                values = np.asarray(values)
                summary_rows.append(dict(Dataset=dataset, Benchmark=benchmark, Architecture=architecture, Strategy=strategy,
                                         Fraction=fraction, Delta_mean=values.mean(), Delta_min=values.min(), Delta_max=values.max(),
                                         Delta_std=values.std(ddof=1), **{f'Seed{s}': v for s, v in zip(SEEDS, values)}))
    return pd.DataFrame(seed_rows), pd.DataFrame(summary_rows)


def draw_panel(ax, summary, architecture):
    baseline = ax.axhline(0., color='#368353', linestyle='--', linewidth=1.25, label='Full dataset baseline')
    handles = [baseline]
    for strategy in STRATEGIES.values():
        part = summary[summary.Architecture.eq(architecture) & summary.Strategy.eq(strategy)].sort_values('Fraction')
        require(part.Fraction.tolist() == FRACTIONS, 'Incomplete plot curve')
        x, center, low, high = (part[c].to_numpy() for c in ['Fraction', 'Delta_mean', 'Delta_min', 'Delta_max'])
        band = ax.fill_between(x, low, high, color=COLORS[strategy], alpha=.18, edgecolor='none')
        line, = ax.plot(x, center, color=COLORS[strategy], marker=MARKERS[strategy], markersize=5.7,
                        markeredgecolor='white', markeredgewidth=.55, linewidth=1.45, label=strategy)
        require(np.array_equal(line.get_ydata(), center), 'Drawn mean differs from numeric data')
        vertices = np.concatenate([path.vertices for path in band.get_paths()])
        for fraction, minimum, maximum in zip(x, low, high):
            actual = vertices[np.isclose(vertices[:, 0], fraction, rtol=0, atol=1e-14), 1]
            require(actual.min() == minimum and actual.max() == maximum, 'Drawn band is not the seed min-max')
        handles.append(line)
    ax.set_xlabel('Retained training subset')
    ax.set_xticks(FRACTIONS, ['10%', '20%', '30%'])
    ax.set_xlim(.085, .315)
    ax.grid(axis='y', color='#D8D8D8', linewidth=.55)
    ax.grid(axis='x', color='#E5E5E5', linewidth=.45, linestyle=':')
    ax.set_axisbelow(True)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    return handles


def plot_figure(summary, output, basename, title):
    architectures = [a for a in ['TDnCNN', 'DRUNet'] if a in set(summary.Architecture)]
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'STIXGeneral', 'DejaVu Serif'],
                         'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
                         'axes.labelsize': 11, 'axes.titlesize': 11.5, 'legend.fontsize': 9.5})
    fig, axes = plt.subplots(1, len(architectures), figsize=(3.6*len(architectures), 3.6), sharey=True, squeeze=False)
    low, high = min(summary.Delta_min.min(), 0.), max(summary.Delta_max.max(), 0.)
    pad = max((high-low)*.08, .08)
    limits = (math.floor((low-pad)*10)/10, math.ceil((high+pad)*10)/10)
    for index, (ax, architecture) in enumerate(zip(axes[0], architectures)):
        handles = draw_panel(ax, summary, architecture)
        ax.set_title(f'({chr(97+index)}) {architecture}', loc='left')
        ax.set_ylim(*limits)
    axes[0, 0].set_ylabel(r'$\Delta$ PSNR vs Full (dB)')
    fig.suptitle(title)
    fig.legend(handles, ['Full dataset baseline', *STRATEGIES.values()], loc='lower center',
               ncol=2 if len(architectures) == 1 else 4, frameon=False)
    fig.tight_layout(rect=(0, .16, 1, .93))
    paths = []
    for extension in ['png', 'pdf', 'svg']:
        path = output/f'{basename}.{extension}'
        fig.savefig(path, dpi=600, facecolor='white')
        paths.append(path.name)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, default=PROJECT/'results_final')
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory; existing outputs are never overwritten.')
    parser.add_argument('--scope', choices=['internal', 'external', 'all'], default='internal')
    parser.add_argument('--dataset', choices=['all', *DATASETS], default='all')
    parser.add_argument('--verify-source-artifacts',action='store_true',help='Also reopen original per-image sources/checkpoints; otherwise require verified final-table manifests.')
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'Choose a new output directory')
    require(args.scope == 'internal' or args.dataset in ['all', 'imagenet100'], 'Final external scope is ImageNet-100 DRUNet only')
    frames, sources = [], []
    if args.scope in ['internal', 'all']:
        base = args.results_root/'tables_3seeds'
        raw, means, *_ = validate_internal(base, verify_sources=args.verify_source_artifacts)
        raw = raw.copy();raw['Benchmark'] = 'internal'
        if args.dataset != 'all':
            raw = raw[raw.Dataset.eq(DATASETS[args.dataset])]
        frames.append(raw)
        sources += [base/'final_results_all_seeds.csv', base/'final_results_mean_3seeds.csv']
    if args.scope in ['external', 'all']:
        base = args.results_root/'imagenet100_external'
        raw, means, _ = validate_external(base, verify_sources=args.verify_source_artifacts)
        raw = raw.rename(columns={'Training corpus seed': 'Seed'}).copy()
        raw['Dataset'] = 'ImageNet-100'
        frames.append(raw)
        sources += [base/'imagenet100_external_all_seeds.csv', base/'imagenet100_external_mean_3seeds.csv']
    seeds, summary = paired_deltas(pd.concat(frames, ignore_index=True))
    args.output_dir.mkdir(parents=True)
    seeds.to_csv(args.output_dir/'paired_deltas.csv', index=False)
    summary.to_csv(args.output_dir/'plot_summary.csv', index=False)
    reloaded = pd.read_csv(args.output_dir/'plot_summary.csv')
    require(np.allclose(reloaded.select_dtypes('number'), summary.select_dtypes('number'), rtol=0, atol=1e-14), 'CSV precision drift')
    figures = []
    for (dataset, benchmark), group in summary.groupby(['Dataset', 'Benchmark'], sort=False):
        key = next(key for key, label in DATASETS.items() if label == dataset)
        suffix = 'internal' if benchmark == 'internal' else benchmark.lower()
        figures += plot_figure(group, args.output_dir, f'{key}_{suffix}_delta_psnr_3seeds', dataset if benchmark == 'internal' else f'{dataset} / {benchmark}')
    evidence = dict(status='PASS', seeds=SEEDS, central_line='mean of same-seed paired deltas', band='min-max across seeds',
                    seed_rows=len(seeds), summary_points=len(summary), files=figures, source_validation='full source artifacts' if args.verify_source_artifacts else 'verified final aggregation manifests',
                    source_sha256={p.relative_to(args.results_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, training=False, inference=False)
    (args.output_dir/'plot_validation.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='PASS', figures=len(figures)//3, paired_rows=len(seeds), summary_points=len(summary)), indent=2))


if __name__ == '__main__':
    main()
