"""
A-PARSE-5: Slither feature correlation POC.

Runs Slither's `halstead` and `ck` printers on a representative sample of
compilable contracts (300 contracts, stratified by top solc versions), then
computes Spearman correlations between SCAFE's complexity features and
Slither's IR-derived equivalents.

Goal: validate that SCAFE's regex/AST-based feature extraction produces
metrics that strongly agree with Slither's IR analysis on the same contracts.
"""
import sys, os, re, subprocess, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

STRICT_CSV   = "data/derived/unified_contract_eval_scored_strict.csv"
SLITHER_CSV  = "data/derived/slither_results/slither_per_contract.csv"
OUT_DIR      = Path("data/derived/slither_correlation")
SAMPLE_N     = 300   # contracts to run Slither on
RANDOM_SEED  = 42

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_table(lines, start):
    """Parse a simple ASCII table returned by Slither printers.
    Returns list of dicts with column names as keys."""
    rows = []
    header = None
    for i in range(start, min(start + 50, len(lines))):
        line = lines[i].strip()
        if not line or set(line) <= set('+-'):
            if header is not None and rows:
                break
            continue
        cells = [c.strip() for c in line.strip('|').split('|')]
        if header is None:
            header = cells
        else:
            rows.append(dict(zip(header, cells)))
    return rows


def parse_halstead(output: str) -> dict:
    """Extract per-contract Halstead values from Slither output."""
    lines = output.splitlines()
    result = {}
    for i, line in enumerate(lines):
        if 'Total Operators' in line and 'Total Operands' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    for k in ['Total Operators', 'Unique Operators', 'Total Operands', 'Unique Operands']:
                        try:
                            result[name][k] = int(r.get(k, 0) or 0)
                        except (ValueError, TypeError):
                            result[name][k] = 0
        if 'Vocabulary' in line and 'Volume' in line and 'Program Length' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    for k in ['Vocabulary', 'Program Length', 'Volume']:
                        try:
                            result[name][k] = float(r.get(k, 0) or 0)
                        except (ValueError, TypeError):
                            result[name][k] = 0.0
        if 'Difficulty' in line and 'Effort' in line and 'Time' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    for k in ['Difficulty', 'Effort']:
                        try:
                            result[name][k] = float(r.get(k, 0) or 0)
                        except (ValueError, TypeError):
                            result[name][k] = 0.0
    # Aggregate across contracts in the file (sum)
    agg = {'slither_halstead_volume': 0.0, 'slither_halstead_difficulty': 0.0,
           'slither_halstead_effort': 0.0, 'slither_total_operators': 0,
           'slither_total_operands': 0, 'slither_n_contracts': len(result)}
    for r in result.values():
        agg['slither_halstead_volume']     += r.get('Volume', 0)
        agg['slither_halstead_difficulty'] += r.get('Difficulty', 0)
        agg['slither_halstead_effort']     += r.get('Effort', 0)
        agg['slither_total_operators']     += r.get('Total Operators', 0)
        agg['slither_total_operands']      += r.get('Total Operands', 0)
    return agg


def parse_ck(output: str) -> dict:
    """Extract CK metrics from Slither output."""
    lines = output.splitlines()
    result = {}
    for i, line in enumerate(lines):
        if 'State variables' in line and 'Constants' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    try:
                        result[name]['state_vars'] = int(r.get('State variables', 0) or 0)
                    except (ValueError, TypeError):
                        result[name]['state_vars'] = 0
        if 'Public' in line and 'External' in line and 'Internal' in line and 'Private' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    for k in ['Public', 'External', 'Internal', 'Private']:
                        try:
                            result[name][k.lower() + '_fns'] = int(r.get(k, 0) or 0)
                        except (ValueError, TypeError):
                            result[name][k.lower() + '_fns'] = 0
        if 'Mutating' in line and 'View' in line and 'Pure' in line:
            rows = _parse_table(lines, i)
            for r in rows:
                name = r.get('Contract', '').strip()
                if name and name.lower() != 'total':
                    result.setdefault(name, {})
                    for k in ['Mutating', 'View', 'Pure']:
                        try:
                            result[name][k.lower() + '_fns'] = int(r.get(k, 0) or 0)
                        except (ValueError, TypeError):
                            result[name][k.lower() + '_fns'] = 0
    agg = {'slither_n_functions': 0, 'slither_state_vars': 0,
           'slither_public_fns': 0, 'slither_external_fns': 0,
           'slither_view_pure_fns': 0}
    for r in result.values():
        fns = r.get('public_fns', 0) + r.get('external_fns', 0) + r.get('internal_fns', 0) + r.get('private_fns', 0)
        agg['slither_n_functions']  += fns
        agg['slither_state_vars']   += r.get('state_vars', 0)
        agg['slither_public_fns']   += r.get('public_fns', 0)
        agg['slither_external_fns'] += r.get('external_fns', 0)
        agg['slither_view_pure_fns'] += r.get('view_fns', 0) + r.get('pure_fns', 0)
    return agg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_slither_on_file(sol_path: str, solc_version: str, debug: bool = False) -> dict | None:
    """Run Slither halstead+ck on a single .sol file; return aggregated dict."""
    try:
        solc_bin = str(Path(
            os.path.expanduser("~"),
            "PycharmProjects/sc_review/venv/.solc-select/artifacts",
            f"solc-{solc_version}", f"solc-{solc_version}"
        ))
        if not os.path.exists(solc_bin):
            if debug:
                print(f"    MISSING solc {solc_version} at {solc_bin}", flush=True)
            return None

        env = os.environ.copy()
        env['SOLC_VERSION'] = solc_version

        def run_printer(printer):
            r = subprocess.run(
                ["slither", sol_path, "--print", printer,
                 "--solc", solc_bin, "--no-fail",
                 "--disable-color"],
                capture_output=True, text=True, timeout=30, env=env
            )
            return r.stdout + r.stderr

        h_out = run_printer("halstead")
        c_out = run_printer("ck")

        if debug:
            print(f"    h_out[:200]: {h_out[:200]}", flush=True)

        h_metrics = parse_halstead(h_out)
        c_metrics = parse_ck(c_out)

        if debug:
            print(f"    h_metrics: {h_metrics}", flush=True)
            print(f"    c_metrics: {c_metrics}", flush=True)

        if h_metrics['slither_n_contracts'] == 0:
            return None

        return {**h_metrics, **c_metrics, 'sol_path': sol_path}
    except subprocess.TimeoutExpired:
        if debug:
            print(f"    TIMEOUT: {sol_path[-50:]}", flush=True)
        return None
    except Exception as e:
        print(f"    ERROR {sol_path[-50:]}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return None


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    sdf = pd.read_csv(SLITHER_CSV)
    udf = pd.read_csv(STRICT_CSV, low_memory=False)

    # Only contracts in strict dataset, excluding paths with glob-unsafe chars
    strict_paths = set(udf['local_source_path'].dropna())
    compilable = sdf[
        sdf['slither_success'] == True
    ].copy()
    compilable = compilable[compilable['sol_path'].isin(strict_paths)]
    # Drop paths with [ ] which confuse Slither's internal glob expansion
    compilable = compilable[~compilable['sol_path'].str.contains(r'\[|\]', regex=True)]
    print(f"Compilable contracts in strict dataset (bracket-free): {len(compilable)}")

    # Stratified sample across top versions
    top_versions = compilable['solc_version'].value_counts().head(6).index.tolist()
    sample_per_ver = max(1, SAMPLE_N // len(top_versions))
    rng = np.random.default_rng(RANDOM_SEED)

    sampled_rows = []
    for v in top_versions:
        subset = compilable[compilable['solc_version'] == v]
        n = min(sample_per_ver, len(subset))
        idx = rng.choice(len(subset), size=n, replace=False)
        sampled_rows.append(subset.iloc[idx])
    sample = pd.concat(sampled_rows).reset_index(drop=True)
    print(f"Sampled {len(sample)} contracts across {len(top_versions)} solc versions")

    # Run Slither
    results = []
    t0 = time.time()
    for i, row in sample.iterrows():
        if i % 20 == 0:
            elapsed = time.time() - t0
            eta = (elapsed / max(i, 1)) * (len(sample) - i)
            print(f"  [{i}/{len(sample)}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
        metrics = run_slither_on_file(row['sol_path'], row['solc_version'])
        if metrics is not None:
            results.append(metrics)

    print(f"\nSlither succeeded on {len(results)}/{len(sample)} contracts")
    if not results:
        print("ERROR: No contracts processed successfully. Check Slither output.")
        sys.exit(1)
    slither_df = pd.DataFrame(results)
    slither_df.to_csv(OUT_DIR / "slither_metrics_sample.csv", index=False)

    # Join with SCAFE features
    scafe_cols = [
        'local_source_path', 'loc', 'n_functions', 'n_public_external_functions',
        'n_payable_functions', 'halstead_volume', 'halstead_difficulty', 'halstead_effort',
        'cyclomatic_per_loc', 'n_complex_functions', 'n_high_complexity_fns',
        'ast_n_functions', 'avg_loc_per_function', 'max_function_loc',
        'fn_loc_gini', 'fn_loc_cv', 'fn_loc_skew',
    ]
    scafe_sub = udf[[c for c in scafe_cols if c in udf.columns]].copy()
    scafe_sub = scafe_sub.rename(columns={'local_source_path': 'sol_path'})

    merged = slither_df.merge(scafe_sub, on='sol_path', how='inner')
    print(f"Merged rows (Slither ∩ SCAFE): {len(merged)}")
    merged.to_csv(OUT_DIR / "merged_slither_scafe.csv", index=False)

    # Correlation analysis
    pairs = [
        # (SCAFE feature,             Slither feature,                 description)
        ('halstead_volume',           'slither_halstead_volume',       'Halstead Volume'),
        ('halstead_difficulty',       'slither_halstead_difficulty',   'Halstead Difficulty'),
        ('halstead_effort',           'slither_halstead_effort',       'Halstead Effort'),
        ('n_functions',               'slither_n_functions',           'Function Count'),
        ('ast_n_functions',           'slither_n_functions',           'AST Function Count'),
        ('n_public_external_functions','slither_public_fns',           'Public/External Fns'),
        ('loc',                       'slither_total_operators',       'LOC vs Operators (proxy)'),
    ]

    print("\n" + "="*70)
    print(f"{'SCAFE Feature':<30}  {'Slither Feature':<30}  {'Spearman r':>10}  {'p-value':>10}")
    print("="*70)
    corr_rows = []
    for scafe_f, slither_f, desc in pairs:
        if scafe_f not in merged.columns or slither_f not in merged.columns:
            continue
        x = pd.to_numeric(merged[scafe_f], errors='coerce')
        y = pd.to_numeric(merged[slither_f], errors='coerce')
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            continue
        r, p = spearmanr(x[mask], y[mask])
        print(f"  {scafe_f:<30}  {slither_f:<30}  {r:>10.3f}  {p:>10.2e}")
        corr_rows.append({'scafe_feature': scafe_f, 'slither_feature': slither_f,
                          'description': desc, 'spearman_r': round(r, 4),
                          'p_value': p, 'n': int(mask.sum())})
    print("="*70)

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUT_DIR / "correlation_results.csv", index=False)

    print(f"\nResults saved to {OUT_DIR}/")
    print("\nSummary:")
    if not corr_df.empty:
        mean_r = corr_df['spearman_r'].abs().mean()
        max_r  = corr_df['spearman_r'].abs().max()
        high_r = (corr_df['spearman_r'].abs() >= 0.7).sum()
        print(f"  Mean |r|  = {mean_r:.3f}")
        print(f"  Max  |r|  = {max_r:.3f}")
        print(f"  # pairs with |r| ≥ 0.70: {high_r} / {len(corr_df)}")
    return corr_df


if __name__ == "__main__":
    main()
