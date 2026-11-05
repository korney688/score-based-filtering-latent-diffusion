"""Validate saved results, export tables, or wait for evaluation to finish.
"""
from pathlib import Path
import argparse
import csv
import datetime as dt
import hashlib
import itertools
import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import psutil

PROJECT = Path(__file__).resolve().parents[1]
EXTERNAL = PROJECT / 'results_final/imagenet100_external'
INTERNAL = PROJECT / 'results_final/tables_3seeds'
OUT = PROJECT / 'results_final/consultant_report'
SEEDS = [42, 43, 44]
DATASETS = ['Colored MNIST', 'CIFAR-10', 'ImageNet-100']
ARCHS = ['TDnCNN', 'DRUNet']
BENCHMARKS = ['Kodak24', 'CBSD68', 'Urban100', 'SIDD']
VARIANTS = ['full'] + [f'{s}_p{p}' for s in ('topk', 'quantile', 'random') for p in (10, 20, 30)]
METRICS = ['PSNR', 'SSIM', 'LPIPS', 'MSE', 'Delta PSNR vs Full']
REPORTS = ['RESULTS_FOR_CONSULTANT.md', 'INTERNAL_ALL_SEEDS.md', 'EXTERNAL_IMAGENET.md', 'PROTOCOL_AUDIT.md']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value, encoding='utf-8')
    temp.replace(path)


def dump(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def label(variant):
    if variant == 'full':
        return 'Full'
    strategy, percent = variant.split('_p')
    return {'topk': 'Top-K', 'quantile': 'Quantile', 'random': 'Random'}[strategy] + f' {percent}%'


def table(headers, rows):
    def cell(value):
        return str(value).replace('|', '\\|').replace('\n', ' ')
    result = ['| ' + ' | '.join(map(cell, headers)) + ' |',
              '| ' + ' | '.join(['---'] + ['---:'] * (len(headers) - 1)) + ' |']
    result += ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows]
    return '\n'.join(result) + '\n'


def link(path, text=None):
    path = Path(path)
    try:
        relative = os.path.relpath(path, OUT).replace('\\', '/')
    except ValueError:  # Sources may be on another Windows drive.
        relative = path.resolve().as_posix()
    return f'[{text or path.name}](<{relative}>)'


def inventory():
    ready, missing = [], []
    for seed, benchmark, variant in itertools.product(SEEDS, BENCHMARKS, VARIANTS):
        folder = EXTERNAL / f'seed{seed}' / benchmark / variant
        row = dict(Seed=seed, Benchmark=benchmark, Variant=variant)
        destination = ready if all((folder / name).is_file() for name in ('provenance.json', 'per_observation_metrics.csv')) else missing
        destination.append(row)
    return ready, missing


def evaluators():
    found = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        try:
            command = process.info['cmdline'] or []
            if any(Path(arg).name == 'final_imagenet_external.py' for arg in command) and 'evaluate' in command:
                found.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def blocked(reason, missing, active):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'MISSING_EXTERNAL_COMBINATIONS.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['Seed', 'Benchmark', 'Variant'])
        writer.writeheader()
        writer.writerows(missing)
    groups = []
    for seed, benchmark in itertools.product(SEEDS, BENCHMARKS):
        variants = [r['Variant'] for r in missing if r['Seed'] == seed and r['Benchmark'] == benchmark]
        if variants:
            groups.append([seed, benchmark, ', '.join(variants)])
    body = '# CONSULTANT TABLES BLOCKED\n\n' + reason + '\n\n'
    body += f'External combinations: {120-len(missing)}/120. Missing: {len(missing)}. Evaluator PID: {active or "none"}.\n\n'
    body += table(['Seed', 'Benchmark', 'Missing variants'], groups)
    body += '\nFinal Markdown reports and aggregate are not started while evaluation is active or results are incomplete.\n'
    body += '\n[Complete missing list](MISSING_EXTERNAL_COMBINATIONS.csv).\n'
    write(OUT / 'BLOCKED_REPORT.md', body)
    dump(OUT / 'STATUS.json', dict(status='WAITING' if active else 'BLOCKED', reason=reason,
         timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(), evaluator_pids=active,
         external_ready=120-len(missing), external_expected=120, missing=missing,
         training_started=False, inference_started=False, reports_created=False))


def validate_grid(raw, means, dimensions, seed_column, expected_keys, allow_mnist_lpips=False):
    keys = dimensions + [seed_column]
    require(not raw.duplicated(keys).any(), 'Duplicate per-seed result rows')
    require(set(raw[keys].itertuples(index=False, name=None)) == expected_keys, 'Missing or unexpected dataset/model/variant/seed')
    require(not means.duplicated(dimensions).any(), 'Duplicate mean result rows')
    expected_groups = {key[:-1] for key in expected_keys}
    require(set(means[dimensions].itertuples(index=False, name=None)) == expected_groups, 'Missing mean rows')
    for metric in METRICS:
        missing = raw[metric].isna()
        permitted = (raw['Dataset'].eq('Colored MNIST') if allow_mnist_lpips and metric == 'LPIPS' else pd.Series(False, index=raw.index))
        require((missing == permitted).all(), 'Unexpected missing metric: ' + metric)
        require(np.isfinite(raw.loc[~missing, metric]).all(), 'Nonfinite metric: ' + metric)
    base_dimensions = [name for name in dimensions if name != 'Variant'] + [seed_column]
    full = raw[raw.Variant.eq('full')].set_index(base_dimensions).PSNR
    for _, row in raw.iterrows():
        variant = row['Variant'];fraction = 1.0 if variant == 'full' else int(variant.rsplit('p', 1)[1]) / 100
        require(row['Strategy'] == label(variant).split(' ')[0] and np.isclose(row['Fraction'], fraction), 'Strategy/fraction mismatch')
        index = tuple(row[name] for name in base_dimensions)
        delta = row['PSNR'] - full.loc[index]
        require(np.isclose(row['Delta PSNR vs Full'], delta, rtol=1e-10, atol=1e-12), 'Unpaired/wrong Delta PSNR')
    require((raw.loc[raw.Variant.eq('full'), 'Delta PSNR vs Full'] == 0).all(), 'Full delta is not zero')
    indexed = means.set_index(dimensions)
    for group, frame in raw.groupby(dimensions, sort=False):
        require(len(frame) == 3 and set(frame[seed_column]) == set(SEEDS), 'Need exactly three seeds')
        result = indexed.loc[group]
        require(result['Strategy'] == frame.Strategy.iloc[0] and np.isclose(result['Fraction'], frame.Fraction.iloc[0]), 'Mean strategy/fraction mismatch')
        for metric in METRICS:
            name = 'Delta_PSNR' if metric == 'Delta PSNR vs Full' else metric
            values = frame[metric].to_numpy(dtype=float)
            expected = [np.nan, np.nan] if np.isnan(values).all() else [values.mean(), values.std(ddof=1)]
            actual = [result[name + '_mean'], result[name + '_std']]
            require(np.allclose(actual, expected, rtol=1e-9, atol=1e-12, equal_nan=True), 'Mean/std mismatch: ' + str(group) + '/' + metric)


def validate_internal(root=None, verify_sources=True):
    root = Path(root) if root is not None else INTERNAL
    validation = js(root / 'validation_report.json')
    require(validation['status'] == 'FINAL 3-SEED RESULTS READY FOR ARTICLE' and validation['std_ddof'] == 1, 'Internal prior validation missing')
    files = js(root / 'output_files_sha256.json')
    for filename in ('final_results_all_seeds.csv', 'final_results_mean_3seeds.csv', 'final_results_provenance.json', 'protocol_audit.csv'):
        require(sha(root / filename) == files[filename], 'Internal aggregation artifact hash drift: ' + filename)
    raw = pd.read_csv(root / 'final_results_all_seeds.csv')
    means = pd.read_csv(root / 'final_results_mean_3seeds.csv')
    expected = set(itertools.product(DATASETS, ARCHS, VARIANTS, SEEDS))
    validate_grid(raw, means, ['Dataset', 'Architecture', 'Variant'], 'Seed', expected, True)
    provenance = js(root / 'final_results_provenance.json')['runs']
    require(len(provenance) == 9 and {(r['dataset'], r['seed']) for r in provenance} == set(itertools.product(DATASETS, SEEDS)), 'Expected nine unique canonical runs')
    protocols = pd.read_csv(root / 'protocol_audit.csv')
    require(len(protocols) == 9 and not protocols.duplicated(['Dataset', 'Seed']).any(), 'Invalid internal protocol audit')
    require(set(zip(protocols.Dataset, protocols.Seed)) == set(itertools.product(DATASETS, SEEDS)), 'Missing protocol run')
    require((protocols.Status == 'PASS').all() and (protocols.Corpus_seed == protocols.Seed).all(), 'Protocol run not PASS/corpus seed wrong')
    for dataset, group in protocols.groupby('Dataset'):
        for field in ('TDnCNN_protocol', 'DRUNet_protocol', 'Evaluation_protocol', 'Evaluation_seed', 'Evaluation_manifest_hash'):
            require(group[field].nunique() == 1, f'Protocol mismatch: {dataset}/{field}')
    import build_final_experiment_results as canonical
    for run in provenance:
        dataset, seed = run['dataset'], run['seed']
        frame = raw[(raw.Dataset == dataset) & (raw.Seed == seed)]
        protocol = protocols[(protocols.Dataset == dataset) & (protocols.Seed == seed)].iloc[0]
        require(seed == run['corpus_seed'] and protocol.Run_ID == run['run_id'], 'Run ID/corpus seed mismatch')
        if dataset == 'Colored MNIST' and seed == 42:
            require(run['run_id'] == 'colored_mnist_001_seed42_canonical', 'Legacy Colored MNIST seed42')
        config = Path(run['resolved_config_path']);source = Path(run['source_evaluation_csv'])
        require(set(frame['Source SHA256']) == {run['source_evaluation_sha256']}, 'Internal source hash anchor differs')
        if not verify_sources:
            continue
        source_hash = sha(source)
        require(sha(config) == run['resolved_config_sha256'] and source_hash == run['source_evaluation_sha256'], 'Internal source/config drift')
        require(set(frame['Source file']) == {str(source)} and set(frame['Source SHA256']) == {source_hash}, 'Wrong internal source CSV')
        original = canonical.normalized_frame(pd.read_csv(source), dataset, seed, source)
        for metric in ('PSNR', 'SSIM', 'LPIPS', 'MSE'):
            left = frame.set_index(['Architecture', 'Variant'])[metric].sort_index()
            right = original.set_index(['Architecture', 'Variant'])[metric].sort_index()
            require(left.index.equals(right.index) and np.allclose(left, right, equal_nan=True, rtol=1e-10, atol=1e-12), 'Internal source metric mismatch')
    return raw, means, protocols, provenance, validation['warnings']


def aggregate_if_needed():
    paths = [EXTERNAL / name for name in ('imagenet100_external_all_seeds.csv', 'imagenet100_external_mean_3seeds.csv', 'validation.json')]
    marker_time = max(p.stat().st_mtime for p in EXTERNAL.glob('seed*/*/*/provenance.json'))
    if all(p.is_file() for p in paths) and min(p.stat().st_mtime for p in paths) >= marker_time:
        require(js(paths[2])['status'] == 'IMAGENET EXTERNAL FULL GRID READY', 'Existing external validation is not READY')
        return False
    with (OUT / 'aggregate.log').open('a', encoding='utf-8') as log:
        subprocess.run([sys.executable, '-u', str(PROJECT / 'scripts/final_imagenet_external.py'), 'aggregate'],
                       cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return True


def validate_external(root=None, verify_sources=True):
    root = Path(root) if root is not None else EXTERNAL
    raw = pd.read_csv(root / 'imagenet100_external_all_seeds.csv')
    means = pd.read_csv(root / 'imagenet100_external_mean_3seeds.csv')
    means = means.rename(columns={f'{name} {stat}': f'{name.replace("Delta PSNR", "Delta_PSNR")}_{stat}'
                        for name in ('PSNR', 'SSIM', 'LPIPS', 'MSE', 'Delta PSNR') for stat in ('mean', 'std')})
    validate_grid(raw, means, ['Benchmark', 'Architecture', 'Variant'], 'Training corpus seed',
                  set(itertools.product(BENCHMARKS, ['DRUNet'], VARIANTS, SEEDS)))
    validation = js(root / 'validation.json')
    require(validation['rows'] == 120 and validation['mean_rows'] == 40 and validation['std_ddof'] == 1, 'External validation counts/ddof wrong')
    if not verify_sources:
        files=js(root/'output_files_sha256.json')
        for name in ('imagenet100_external_all_seeds.csv','imagenet100_external_mean_3seeds.csv','validation.json'):
            require(sha(root/name)==files[name],'External final-table hash drift: '+name)
        require(validation['status']=='IMAGENET EXTERNAL FULL GRID READY','External aggregation is not READY')
        return raw, means, []
    checkpoints = pd.read_csv(root / 'checkpoint_inventory.csv').set_index(['Training seed', 'Variant'])
    hashes, protocols, observations, sources = {}, set(), {}, []
    for _, row in raw.iterrows():
        seed, variant, benchmark = int(row['Training corpus seed']), row.Variant, row.Benchmark
        folder = root / f'seed{seed}' / benchmark / variant
        source = folder / 'per_observation_metrics.csv';marker = js(folder / 'provenance.json');identity = marker['identity']
        require(Path(row['Source artifact']).resolve() == source.resolve(), 'Legacy/wrong external source')
        require(identity['seed'] == seed and identity['variant'] == variant and identity['benchmark'] == benchmark and identity['architecture'] == 'DRUNet', 'External cross-seed/model/variant')
        require(sha(source) == marker['output_sha256'], 'External per-image CSV drift')
        cp = checkpoints.loc[(seed, variant)];path = Path(row.Checkpoint)
        require(path.resolve() == Path(cp['Local path']).resolve() and cp['Integrity status'] == 'PASS', 'Noncanonical external checkpoint')
        if path not in hashes:
            hashes[path] = sha(path)
        require(hashes[path] == identity['checkpoint_sha256'] == cp['SHA256'], 'External checkpoint hash drift')
        inputs = identity['inputs'];protocols.add(json.dumps(inputs, sort_keys=True))
        require(inputs['sigma'] == 25 and inputs['noise_seed'] == 42 and inputs['sidd_primary_sigma_conditioning'] == 25 and inputs['sidd_added_noise'] is False, 'External noise protocol mismatch')
        values = pd.read_csv(source)
        if benchmark == 'SIDD':
            require(set(values['model']) == {'drunet'} and set(values['variant']) == {variant} and set(values['assumed_sigma_255']) == {25}, 'SIDD block metadata/conditioning mismatch')
        count = 1280 if benchmark == 'SIDD' else {'Kodak24': 24, 'CBSD68': 68, 'Urban100': 100}[benchmark]
        key = 'global_block_index' if benchmark == 'SIDD' else 'Image'
        require(len(values) == count and values[key].nunique() == count, 'External observations incomplete/duplicated')
        ids = frozenset(values[key])
        if benchmark in observations:
            require(ids == observations[benchmark], 'External observation sets differ between models/seeds')
        observations[benchmark] = ids
        for metric in ('PSNR', 'SSIM', 'LPIPS', 'MSE'):
            require(np.isfinite(values[metric]).all() and np.isclose(values[metric].mean(), row[metric], rtol=1e-10, atol=1e-12), 'External source metric mismatch')
        sources.append(dict(Dataset='ImageNet-100 external', Architecture='DRUNet', Seed=seed, Benchmark=benchmark,
                            Variant=variant, Source=str(source), SHA256=marker['output_sha256']))
    require(len(protocols) == 1 and len(hashes) == 30, 'External inputs differ or wrong checkpoint count')
    return raw, means, sources


def plusminus(row, metric):
    digits = {'PSNR': 3, 'Delta_PSNR': 3, 'SSIM': 4, 'LPIPS': 4, 'MSE': 6}[metric]
    mean, std = row[metric + '_mean'], row[metric + '_std']
    return '—' if pd.isna(mean) else f'{mean:.{digits}f} ± {std:.{digits}f}'


def seed_table(raw, means, seed_column, external=False):
    rows = []
    for variant in VARIANTS:
        frame = raw[raw.Variant == variant].set_index(seed_column)
        mean = means[means.Variant == variant].iloc[0]
        values = [label(variant)] + [f'{frame.loc[seed, "PSNR"]:.3f}' for seed in SEEDS]
        values += [f'{mean.PSNR_mean:.3f}', f'{mean.PSNR_std:.3f}']
        if external:
            values += [f'{mean.Delta_PSNR_mean:.3f}']
        rows.append(values)
    return table(['Variant', 'Seed 42', 'Seed 43', 'Seed 44', 'Mean', 'Std'] + (['Mean Delta vs Full'] if external else []), rows)


def render(raw, means, eraw, emean, protocols, provenance, warnings):
    means = means.reset_index(drop=True)
    emean = emean.reset_index(drop=True)
    intro = ('# Results for consultant\n\n## Protocol summary\n\n'
             '- Три corpus-noise seeds: 42, 43, 44. Значения — mean ± выборочная std (ddof=1) по этим трем seeds.\n'
             '- Filtering: Full / Top-K / Quantile-spread / Random; fractions 10%, 20%, 30%. Top-K выбирает минимальный diffusion score.\n'
             '- Internal: Colored MNIST / CIFAR-10 / ImageNet-100, TDnCNN / DRUNet. External: ImageNet-100 DRUNet.\n'
             '- Delta PSNR — среднее трех paired differences относительно Full того же training seed.\n'
             '- Colored MNIST LPIPS не вычислялся в production и обозначен как —.\n\n')
    internal_sections = [];seed_sections = [];overview = []
    for dataset, architecture in itertools.product(DATASETS, ARCHS):
        r = raw[(raw.Dataset == dataset) & (raw.Architecture == architecture)]
        m = means[(means.Dataset == dataset) & (means.Architecture == architecture)].set_index('Variant').loc[VARIANTS].reset_index()
        rows = [[label(row.Variant)] + [plusminus(row, metric) for metric in ('PSNR', 'Delta_PSNR', 'SSIM', 'LPIPS', 'MSE')] for _, row in m.iterrows()]
        title = f'### {dataset} — {architecture}\n\n'
        internal_sections.append(title + table(['Variant', 'PSNR mean ± std', 'Delta PSNR vs Full mean ± std', 'SSIM mean ± std', 'LPIPS mean ± std', 'MSE mean ± std'], rows)
                                 + '\nSeed-wise PSNR:\n\n' + seed_table(r, m, 'Seed'))
        seed_sections.append(title + seed_table(r, m, 'Seed'))
        overview.append([dataset, architecture] + [plusminus(row, 'PSNR') for _, row in m.iterrows()])
    wide_headers = ['Dataset', 'Model'] + [label(v) for v in VARIANTS]
    internal_overview = '## Internal PSNR summary\n\n' + table(wide_headers, overview)
    external_rows = []
    for benchmark in BENCHMARKS:
        m = emean[emean.Benchmark == benchmark].set_index('Variant')
        external_rows.append([benchmark] + [plusminus(m.loc[v], 'PSNR') for v in VARIANTS])
    external = '## ImageNet-100 DRUNet — external benchmarks\n\n' + table(['Benchmark'] + [label(v) for v in VARIANTS], external_rows)
    external += '\nKodak24 / CBSD68 / Urban100: AWGN sigma=25/255, fixed evaluation noise seed=42.\n\n'
    external_detail = [external]
    for benchmark in BENCHMARKS:
        r = eraw[eraw.Benchmark == benchmark];m = emean[emean.Benchmark == benchmark]
        section = f'### {benchmark}\n\n' + seed_table(r, m, 'Training corpus seed', True)
        if benchmark == 'SIDD':
            section += '\nSIDD использует native real noisy/GT Validation sRGB pairs; дополнительный AWGN не добавляется. DRUNet получает assumed noise-level conditioning sigma=25/255.\n'
        external_detail.append(section)
    extra_external = []
    for benchmark in BENCHMARKS:
        for metric in ('SSIM', 'LPIPS', 'MSE'):
            m = emean[emean.Benchmark == benchmark].set_index('Variant')
            extra_external.append([benchmark, metric] + [plusminus(m.loc[v], metric) for v in VARIANTS])
    external_all = '\n'.join(external_detail) + '\n## External SSIM / LPIPS / MSE\n\n' + table(['Benchmark', 'Metric'] + [label(v) for v in VARIANTS], extra_external)
    observations = []
    for dataset, architecture in itertools.product(DATASETS, ARCHS):
        m = means[(means.Dataset == dataset) & (means.Architecture == architecture)]
        best = m.loc[m.PSNR_mean.idxmax()]
        better = m[m.Variant.str.startswith('topk_') & (m.Delta_PSNR_mean > 0)]
        names = ', '.join(label(v) for v in better.Variant) or 'нет'
        observations.append(f'{dataset} / {architecture}: максимальный mean PSNR — {label(best.Variant)} ({best.PSNR_mean:.3f} dB); Top-K выше Full по среднему paired Delta: {names}.')
    observations.append('Максимальный mean PSNR на external benchmarks: ' + '; '.join(f'{b} — {label(emean[emean.Benchmark == b].loc[emean[emean.Benchmark == b].PSNR_mean.idxmax()].Variant)}' for b in BENCHMARKS) + '.')
    internal_spread = means.loc[means.PSNR_std.idxmax()];external_spread = emean.loc[emean.PSNR_std.idxmax()]
    observations.append(f'Наибольшая internal PSNR std: {internal_spread.Dataset} / {internal_spread.Architecture} / {label(internal_spread.Variant)} — {internal_spread.PSNR_std:.3f} dB.')
    observations.append(f'Наибольшая external PSNR std: {external_spread.Benchmark} / {label(external_spread.Variant)} — {external_spread.PSNR_std:.3f} dB.')
    protocol_rows = []
    for dataset, seed in itertools.product(DATASETS, SEEDS):
        p = protocols[(protocols.Dataset == dataset) & (protocols.Seed == seed)].iloc[0]
        run = next(r for r in provenance if r['dataset'] == dataset and r['seed'] == seed)
        protocol_rows.append([dataset, seed, p.Run_ID, p.Corpus_seed, 'PASS', link(run['source_evaluation_csv'])])
    for seed in SEEDS:
        p = protocols[(protocols.Dataset == 'ImageNet-100') & (protocols.Seed == seed)].iloc[0]
        protocol_rows.append(['ImageNet-100 external', seed, p.Run_ID, seed, 'PASS: 40/40', link(EXTERNAL / 'imagenet100_external_all_seeds.csv')])
    protocol_doc = '# Protocol audit\n\n' + table(['Dataset', 'Seed', 'Run ID', 'Corpus seed', 'Status', 'Source result file'], protocol_rows)
    protocol_doc += '\nAll per-result source paths and SHA256: [SOURCE_FILES.csv](SOURCE_FILES.csv).\n\n'
    protocol_doc += 'Internal protocol snapshot notes:\n\n' + '\n'.join('- ' + warning for warning in warnings) + '\n'
    source_links = '\n## Sources\n\n' + '\n'.join('- ' + link(p) for p in [INTERNAL / 'final_results_all_seeds.csv', INTERNAL / 'final_results_mean_3seeds.csv', EXTERNAL / 'imagenet100_external_all_seeds.csv', EXTERNAL / 'imagenet100_external_mean_3seeds.csv']) + '\n'
    source_links += '\n[Protocol audit](PROTOCOL_AUDIT.md); [per-result source files](SOURCE_FILES.csv); [validation](VALIDATION.json).\n'
    main = intro + '## Internal results\n\n' + '\n'.join(internal_sections) + '\n' + internal_overview + '\n' + '\n'.join(external_detail)
    main += '\n## Main observations\n\n' + '\n'.join('- ' + observation for observation in observations) + '\n' + source_links
    return dict(zip(REPORTS, [main, '# Internal all seeds — PSNR\n\n' + '\n'.join(seed_sections) + source_links,
                              '# ImageNet-100 external results\n\n' + external_all + source_links, protocol_doc]))


def build():
    ready, missing = inventory();active = evaluators()
    require(not missing and not active, 'External evaluation is active or incomplete; no final assembly')
    require(js(EXTERNAL / 'status.json')['status'] == 'COMPLETE', 'External evaluator did not finish successfully')
    import msvcrt
    # Wait until evaluation and aggregation release the lock.
    with (EXTERNAL / '.evaluation.lock').open('r+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            aggregated = aggregate_if_needed()
            raw, means, protocols, provenance, warnings = validate_internal()
            eraw, emean, sources = validate_external()
            documents = render(raw, means, eraw, emean, protocols, provenance, warnings)
            for name, contents in documents.items():
                require(not (OUT / name).exists() or (OUT / name).read_text(encoding='utf-8') == contents, 'Refusing to overwrite different final report: ' + name)
            sources = [dict(Dataset=r.Dataset, Architecture=r.Architecture, Seed=int(r.Seed), Benchmark='internal',
                            Variant=r.Variant, Source=r['Source file'], SHA256=r['Source SHA256']) for _, r in raw.iterrows()] + sources
            pd.DataFrame(sources).to_csv(OUT / 'SOURCE_FILES.csv', index=False)
            validation = dict(status='CONSULTANT TABLES READY', internal_combinations=180, internal_mean_rows=60,
                 external_combinations=120, external_mean_rows=40, seeds=SEEDS, std_ddof=1,
                 timestamp=dt.datetime.now(dt.timezone.utc).isoformat(), aggregate_started_by_builder=aggregated,
                 checks={name: 'PASS' for name in ['expected grids', 'three seeds per group', 'Full present', 'paired deltas',
                    'Full delta zero', 'mean and std from raw seeds', 'finite PSNR', 'all ten variants', 'SIDD present',
                    'canonical Colored MNIST seed42', 'no legacy single-seed Table6 source', 'source CSV/config hashes',
                    'checkpoint hashes', 'same external input protocol and observations', 'source paths retained']},
                 missing_metrics={'Colored MNIST LPIPS': 'Not computed in production; displayed as —'},
                 filtering_sanity='Omitted: medians and class coverage are not available in one unambiguous final summary for all datasets/seeds.',
                 training_started=False, inference_started=False)
            for name, contents in documents.items():
                write(OUT / name, contents)
            validation['report_sha256'] = {name: sha(OUT / name) for name in REPORTS}
            dump(OUT / 'VALIDATION.json', validation)
            dump(OUT / 'STATUS.json', dict(status='CONSULTANT TABLES READY', internal_combinations=180, external_combinations=120,
                 seeds=SEEDS, reports=REPORTS, timestamp=validation['timestamp'], training_started=False, inference_started=False))
            (OUT / 'BLOCKED_REPORT.md').unlink(missing_ok=True)
            print('CONSULTANT TABLES READY', flush=True)
        finally:
            lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def run(wait=False, interval=30):
    OUT.mkdir(parents=True, exist_ok=True)
    import msvcrt
    with (OUT / '.builder.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        while True:
            ready, missing = inventory();active = evaluators()
            if missing or active:
                reason = 'Waiting for existing external evaluator; final assembly has not started.' if active else 'Evaluator is not running and external results are incomplete. Inference will not be restarted.'
                blocked(reason, missing, active)
                print(f'WAIT external={len(ready)}/120 missing={len(missing)} evaluator={active}', flush=True)
                if wait and active:
                    time.sleep(interval)
                    continue
                return 2
            try:
                build()
                return 0
            except Exception as error:
                blocked(str(error), missing, [])
                print('CONSULTANT TABLES BLOCKED:', str(error), flush=True)
                return 2


def export_publication(results_root, output):
    """Export numerical tables without internal source paths or checkpoint inventory."""
    require(not output.exists(), 'Choose a new export directory')
    raw, means, *_ = validate_internal(results_root / 'tables_3seeds', verify_sources=False)
    eraw, emean, _ = validate_external(results_root / 'imagenet100_external', verify_sources=False)
    dimensions = ['Architecture', 'Variant', 'Strategy', 'Fraction']
    mean_columns = [name + '_' + stat for name in ('PSNR', 'SSIM', 'LPIPS', 'MSE', 'Delta_PSNR') for stat in ('mean', 'std')]
    # Export only the listed columns to keep internal paths private.
    tables = {
        'internal_all_seeds.csv': raw[['Dataset', 'Seed', *dimensions, *METRICS]],
        'internal_mean_3seeds.csv': means[['Dataset', *dimensions, *mean_columns]],
        'external_all_seeds.csv': eraw[['Benchmark', 'Training corpus seed', *dimensions, *METRICS]],
        'external_mean_3seeds.csv': emean[['Benchmark', *dimensions, *mean_columns]],
    }
    output.mkdir(parents=True)
    for name, frame in tables.items():
        frame.to_csv(output / name, index=False)
    body = ('# Publication results\n\n'
            'Corpus seeds: 42, 43, 44. Sample standard deviation: ddof=1. '
            'Delta PSNR pairs each branch with Full of the same seed. '
            'Missing Colored MNIST LPIPS remains missing.\n\n'
            'Synthetic external evaluation: AWGN25/255, evaluation seed42. '
            'SIDD: native noisy/GT pairs, no added AWGN; fixed DRUNet conditioning25/255.\n\n')
    for name, frame in tables.items():
        body += f'- [{name}]({name})\n'
        if 'mean_3seeds' in name:
            columns = [frame.columns[0], 'Architecture', 'Variant', 'PSNR_mean', 'PSNR_std', 'Delta_PSNR_mean', 'Delta_PSNR_std']
            body += '\n' + table(columns, frame[columns].itertuples(index=False, name=None)) + '\n'
    write(output / 'RESULTS_FOR_CONSULTANT.md', body)
    source_names = ['tables_3seeds/final_results_all_seeds.csv', 'tables_3seeds/final_results_mean_3seeds.csv',
                    'imagenet100_external/imagenet100_external_all_seeds.csv', 'imagenet100_external/imagenet100_external_mean_3seeds.csv']
    dump(output / 'export_manifest.json', dict(
        status='PASS', seeds=SEEDS, std_ddof=1, source_validation='verified final aggregation manifests',
        source_sha256={name: sha(results_root / name) for name in source_names},
        files_sha256={name: sha(output / name) for name in [*tables, 'RESULTS_FOR_CONSULTANT.md']},
        internal_rows=len(raw), external_rows=len(eraw), training=False, inference=False,
        note='Sanitized numerical export; internal provenance is retained separately. This is not an input bundle for canonical validators.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['wait', 'build', 'status', 'validate', 'export-publication'])
    parser.add_argument('--results-root',type=Path,default=None,help='Read-only input root for validate/export-publication.')
    parser.add_argument('--output-dir',type=Path,default=None,help='New directory for export-publication only.')
    args = parser.parse_args()
    if args.action == 'export-publication':
        if args.output_dir is None:
            parser.error('export-publication requires --output-dir')
        export_publication(args.results_root or PROJECT / 'results_final', args.output_dir)
        print('Publication export ready')
        return 0
    if args.output_dir is not None:
        parser.error('--output-dir is supported only by export-publication')
    if args.action == 'validate':
        root=args.results_root or PROJECT/'results_final'
        raw, means, *_=validate_internal(root/'tables_3seeds')
        external, external_means, _=validate_external(root/'imagenet100_external')
        print(json.dumps(dict(status='PASS',internal_rows=len(raw),internal_means=len(means),external_rows=len(external),external_means=len(external_means))))
        return 0
    if args.results_root is not None:
        parser.error('--results-root is supported only by validate/export-publication')
    if args.action == 'status':
        ready, missing = inventory()
        print(json.dumps(dict(external_ready=len(ready), missing=missing, evaluator_pids=evaluators()), indent=2))
        return 0
    return run(wait=args.action == 'wait')


if __name__ == '__main__':
    sys.exit(main())
