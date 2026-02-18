"""Deep non-compilation features exploration — call graph, data flow, AST, vuln templates."""
import pandas as pd, numpy as np, re, sys
from pathlib import Path
from collections import Counter, defaultdict
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

scored = pd.read_csv(REPO / "data/derived/unified_contract_eval_scored.csv", low_memory=False)
dapp = scored[scored["source_dataset"]=="DAppSCAN"].copy()
dapp["label_status"] = dapp["label_status"].fillna("").astype(str)
dapp = dapp[dapp["label_status"].isin(["clean","vulnerable"])].copy()

vuln = dapp[dapp["label_status"]=="vulnerable"].sample(n=min(500, dapp["label_status"].eq("vulnerable").sum()), random_state=42)
clean = dapp[dapp["label_status"]=="clean"].sample(n=500, random_state=42)

def read_sol(row):
    p = str(row.get("local_source_path","") or "").strip()
    if not p: return ""
    pp = Path(p)
    if pp.is_dir():
        texts = []
        for s in sorted(pp.rglob("*.sol"))[:30]:
            try: texts.append(s.read_text(encoding="utf-8",errors="ignore"))
            except: pass
        return "\n".join(texts)
    elif pp.is_file():
        try: return pp.read_text(encoding="utf-8",errors="ignore")
        except: return ""
    return ""

try:
    from solidity_parser import parser as sol_parser
    HAS_PARSER = True
except:
    HAS_PARSER = False

def _get(node, key, default=None):
    if isinstance(node, dict): return node.get(key, default)
    return getattr(node, key, default)

def _walk_stmts(node, depth=0):
    if depth > 10 or node is None: return []
    stmts = []
    if isinstance(node, list):
        for n in node: stmts.extend(_walk_stmts(n, depth+1))
    elif isinstance(node, dict) or hasattr(node, 'type'):
        stmts.append(node)
        for key in ["statements","body","trueBody","falseBody","expression","subExpression"]:
            child = _get(node, key)
            if child: stmts.extend(_walk_stmts(child, depth+1))
    return stmts

AST_KEYS = ["ast_n_contracts","ast_n_interfaces","ast_n_libraries",
            "ast_n_functions","ast_inheritance_edges",
            "ast_n_public_fns","ast_n_external_fns","ast_public_ratio",
            "ast_n_view_fns","ast_n_pure_fns","ast_n_payable_fns",
            "ast_pure_view_ratio","ast_payable_ratio",
            "ast_max_params","ast_avg_params",
            "ast_max_fn_stmts","ast_avg_fn_stmts","ast_std_fn_stmts"]

def ast_features(text):
    f = {}
    if not HAS_PARSER or not text:
        return {k:0 for k in AST_KEYS}
    try:
        ast = sol_parser.parse(text, loc=True)
    except:
        return {k:0 for k in AST_KEYS}

    children = ast.get("children",[]) if isinstance(ast,dict) else getattr(ast,"children",[])
    contracts = [c for c in children if _get(c,"type")=="ContractDefinition"]

    f["ast_n_contracts"] = len(contracts)
    f["ast_n_interfaces"] = sum(1 for c in contracts if _get(c,"kind") in ("interface",))
    f["ast_n_libraries"] = sum(1 for c in contracts if _get(c,"kind") in ("library",))

    all_fns=[]; fn_vis=Counter(); fn_mut=Counter()
    fn_complexities=[]; fn_params_list=[]; inheritance=0

    for contract in contracts:
        bases = _get(contract, "baseContracts", [])
        inheritance += len(bases) if isinstance(bases, list) else 0
        for node in _get(contract, "subNodes", []):
            ntype = _get(node, "type")
            if ntype == "FunctionDefinition":
                all_fns.append(node)
                vis = _get(node, "visibility", "default")
                fn_vis[vis] += 1
                mut = _get(node, "stateMutability")
                if mut: fn_mut[mut] += 1
                body = _get(node, "body")
                if body:
                    fn_complexities.append(len(_walk_stmts(body)))
                params = _get(node, "parameters", {})
                pl = _get(params, "parameters", []) if isinstance(params, dict) else getattr(params, "parameters", []) or []
                fn_params_list.append(len(pl) if isinstance(pl, list) else 0)

    n_fns = max(1, len(all_fns))
    f["ast_n_functions"] = len(all_fns)
    f["ast_inheritance_edges"] = inheritance
    f["ast_n_public_fns"] = fn_vis.get("public",0)
    f["ast_n_external_fns"] = fn_vis.get("external",0)
    f["ast_public_ratio"] = fn_vis.get("public",0) / n_fns
    f["ast_n_view_fns"] = fn_mut.get("view",0)
    f["ast_n_pure_fns"] = fn_mut.get("pure",0)
    f["ast_n_payable_fns"] = fn_mut.get("payable",0)
    f["ast_pure_view_ratio"] = (fn_mut.get("view",0)+fn_mut.get("pure",0)) / n_fns
    f["ast_payable_ratio"] = fn_mut.get("payable",0) / n_fns
    if fn_params_list:
        f["ast_max_params"] = max(fn_params_list)
        f["ast_avg_params"] = float(np.mean(fn_params_list))
    else:
        f["ast_max_params"] = 0; f["ast_avg_params"] = 0
    if fn_complexities:
        f["ast_max_fn_stmts"] = max(fn_complexities)
        f["ast_avg_fn_stmts"] = float(np.mean(fn_complexities))
        f["ast_std_fn_stmts"] = float(np.std(fn_complexities))
    else:
        f["ast_max_fn_stmts"] = 0; f["ast_avg_fn_stmts"] = 0; f["ast_std_fn_stmts"] = 0
    return f


DEEP_KEYS = ["cg_n_edges","cg_density","cg_max_out_degree","cg_max_in_degree",
    "cg_n_leaf_fns","cg_leaf_ratio","cg_max_call_depth",
    "df_n_shared_write_vars","df_shared_write_ratio","df_n_self_mutate_fns",
    "df_self_mutate_ratio","df_n_raw_pairs",
    "vt_reentrancy_sites","vt_unchecked_low_level","vt_delegatecall_count",
    "vt_tx_origin_auth","vt_missing_zero_check",
    "sp_cei_compliant_fns","sp_cei_ratio","sp_access_control_coverage",
    "sp_event_on_state_change"]

def deep_regex_features(text):
    f = {}
    if not text: return {k:0 for k in DEEP_KEYS}

    fn_names = set(re.findall(r'\bfunction\s+(\w+)\s*\(', text))
    _FB = re.compile(r'function\s+(\w+)\s*\([^)]*\)[^{;]*\{((?:[^{}]|\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)\}', re.DOTALL)
    func_bodies = _FB.findall(text)
    n_fns = max(1, len(fn_names))

    # Call graph
    call_graph = defaultdict(set)
    reverse_graph = defaultdict(set)
    for fn_name, body in func_bodies:
        for other_fn in fn_names:
            if other_fn != fn_name and re.search(rf'\b{re.escape(other_fn)}\s*\(', body):
                call_graph[fn_name].add(other_fn)
                reverse_graph[other_fn].add(fn_name)

    n_edges = sum(len(v) for v in call_graph.values())
    f["cg_n_edges"] = n_edges
    f["cg_density"] = n_edges / max(1, n_fns*(n_fns-1)) if n_fns > 1 else 0
    f["cg_max_out_degree"] = max((len(v) for v in call_graph.values()), default=0)
    f["cg_max_in_degree"] = max((len(v) for v in reverse_graph.values()), default=0)
    f["cg_n_leaf_fns"] = sum(1 for fn in fn_names if not call_graph.get(fn))
    f["cg_leaf_ratio"] = f["cg_n_leaf_fns"] / n_fns
    max_depth = 0
    for root in fn_names:
        if reverse_graph.get(root): continue
        visited=set(); queue=[(root,0)]
        while queue:
            node,d = queue.pop(0)
            if node in visited: continue
            visited.add(node); max_depth=max(max_depth,d)
            if d > 10: break
            for c in call_graph.get(node,[]): queue.append((c,d+1))
    f["cg_max_call_depth"] = max_depth

    # Data flow
    state_vars = set(re.findall(
        r'\b(?:uint|int|bool|address|bytes|string|mapping)\w*\s+'
        r'(?:public\s+|private\s+|internal\s+|constant\s+|immutable\s+)*(\w+)\s*[;=]', text))

    fn_writes = defaultdict(set)
    fn_reads = defaultdict(set)
    for fn_name, body in func_bodies:
        for sv in state_vars:
            if re.search(rf'\b{re.escape(sv)}\s*=', body):
                fn_writes[fn_name].add(sv)
            elif sv in body:
                fn_reads[fn_name].add(sv)

    write_counts = Counter()
    for ws in fn_writes.values(): write_counts.update(ws)
    f["df_n_shared_write_vars"] = sum(1 for v,c in write_counts.items() if c > 1)
    f["df_shared_write_ratio"] = f["df_n_shared_write_vars"] / max(1,len(state_vars))
    f["df_n_self_mutate_fns"] = sum(1 for fn in fn_names if fn_reads.get(fn,set()) & fn_writes.get(fn,set()))
    f["df_self_mutate_ratio"] = f["df_n_self_mutate_fns"] / n_fns

    n_raw = 0
    for fn_a, ws in fn_writes.items():
        for fn_b, rs in fn_reads.items():
            if fn_a != fn_b and ws & rs:
                if fn_b in call_graph.get(fn_a,set()) or fn_a in call_graph.get(fn_b,set()):
                    n_raw += 1
    f["df_n_raw_pairs"] = n_raw

    # Vulnerability templates
    n_reentrancy = 0
    for fn_name, body in func_bodies:
        ext_calls = list(re.finditer(r'\.call\s*[\({]|\.transfer\s*\(|\.send\s*\(', body))
        if not ext_calls: continue
        for sv in state_vars:
            for ec in ext_calls:
                ws = list(re.finditer(rf'\b{re.escape(sv)}\s*=', body))
                if any(w.start() > ec.start() for w in ws):
                    n_reentrancy += 1; break
    f["vt_reentrancy_sites"] = min(n_reentrancy, 50)

    total_calls = len(re.findall(r'\.call\s*[\({]', text))
    checked_calls = len(re.findall(r'\(bool\s+\w+.*?\)\s*=\s*\w+\.call', text))
    f["vt_unchecked_low_level"] = max(0, total_calls - checked_calls)
    f["vt_delegatecall_count"] = len(re.findall(r'\.delegatecall\s*\(', text))
    f["vt_tx_origin_auth"] = len(re.findall(r'require\s*\([^)]*tx\.origin', text))

    n_miss = 0
    for fn_name, body in func_bodies:
        addr_params = re.findall(r'\baddress\s+(\w+)', body[:300])
        for ap in addr_params:
            if not re.search(rf'require\s*\([^)]*{re.escape(ap)}\s*!=\s*address\s*\(0', body):
                n_miss += 1
    f["vt_missing_zero_check"] = n_miss

    # Security patterns
    n_cei = 0
    for fn_name, body in func_bodies:
        reqs = [m.start() for m in re.finditer(r'\brequire\s*\(', body)]
        exts = [m.start() for m in re.finditer(r'\.call[\s{(]|\.transfer\(|\.send\(', body)]
        if reqs and exts and max(reqs) < min(exts):
            n_cei += 1
    f["sp_cei_compliant_fns"] = n_cei
    f["sp_cei_ratio"] = n_cei / n_fns

    acl = 0
    for fn_name, body in func_bodies:
        if re.search(r'\b(public|external)\b', body[:200]):
            if re.search(r'\b(onlyOwner|onlyRole|only\w+|require.*msg\.sender)\b', body):
                acl += 1
    f["sp_access_control_coverage"] = acl / n_fns

    sc_fns = [(fn, body) for fn, body in func_bodies if any(re.search(rf'\b{re.escape(sv)}\s*=', body) for sv in state_vars)]
    f["sp_event_on_state_change"] = sum(1 for _,b in sc_fns if re.search(r'\bemit\s+', b)) / max(1, len(sc_fns))

    return f


# Compute
print("Computing features on 500 vuln + 500 clean contracts...")
results = []
n_ast_ok = 0
for label, subset in [("vulnerable", vuln), ("clean", clean)]:
    for idx, (_, row) in enumerate(subset.iterrows()):
        text = read_sol(row)
        feats = {}
        af = ast_features(text)
        feats.update(af)
        if af.get("ast_n_functions",0) > 0: n_ast_ok += 1
        feats.update(deep_regex_features(text))
        feats["label"] = label
        results.append(feats)
        if (idx+1) % 100 == 0: print(f"  {label}: {idx+1}/{len(subset)}")

rdf = pd.DataFrame(results)
v_df = rdf[rdf["label"]=="vulnerable"]; c_df = rdf[rdf["label"]=="clean"]

print(f"\nAST success: {n_ast_ok}/{len(results)} ({n_ast_ok/len(results)*100:.0f}%)")
print(f"\n{'Feature':<38} {'Vuln':>8} {'Clean':>8} {'AUC':>7} {'Type'}")
print("-"*80)

rows_o = []
for col in sorted(rdf.columns):
    if col=="label": continue
    v = v_df[col].astype(float).values; c = c_df[col].astype(float).values
    y_t = np.concatenate([np.ones(len(v)),np.zeros(len(c))])
    s_t = np.concatenate([v,c])
    try: auc = roc_auc_score(y_t,s_t)
    except: auc = 0.5
    tp = "AST" if col.startswith("ast_") else "CG" if col.startswith("cg_") else "DF" if col.startswith("df_") else "VT" if col.startswith("vt_") else "SP"
    rows_o.append((col, v.mean(), c.mean(), auc, tp))

rows_o.sort(key=lambda x: abs(x[3]-0.5), reverse=True)
for col,vm,cm,auc,tp in rows_o:
    m = "***" if abs(auc-0.5)>0.1 else "**" if abs(auc-0.5)>0.05 else ""
    print(f"  {col:<36} {vm:>8.3f} {cm:>8.3f} {auc:>7.3f} {tp:>5} {m}")
