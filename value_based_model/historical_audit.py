"""Reproduce historical visit-status audits and feature bundles."""
import argparse
import copy
import datetime as dt
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from build_visit_status import digest, write_json
from evaluate_latest_visits import apply_folder_labels, latest_visit_indices
from feature_schema import CLINICAL_VISIT_POLICY, medication_comparison_feature, validate_active_features
from forest_workflow import patient_weights, performance
from longitudinal_features import actual_clinical_visit, day, rows
from sql_snapshot import read_snapshot
from visit_status_workflow import load_status_bundle

ROOT = Path(__file__).resolve().parent
HISTORICAL_SCHEMA = 'recorded_counts_actual_visits_no_coverage_v3'
VARIANTS = ('all_features','selected_features')


def prepare_without_coverage(source_dir, output_dir):
    """Derive the retired feature representation from a checksummed bundle."""
    source, output = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError('Choose a new coverage-free feature bundle directory')
    frames, visits, targets, dictionaries, split, manifest, provenance = load_status_bundle(source)
    source_hash = digest(source/'manifest.json')
    removals = {m:[c for c in X if c.startswith('seizure_coverage_fraction_')] for m,X in frames.items()}
    if not all(removals.values()):
        raise ValueError('Source bundle must include coverage fractions for every active history')
    output.mkdir(parents=True, exist_ok=False)
    for name in ('visits.jsonl','status_targets.jsonl','patient_split.json','eeg_report_audit.jsonl'):
        shutil.copyfile(source/name,output/name)
    for months,X in frames.items():
        reduced=X.drop(columns=removals[months])
        validate_active_features(reduced,months)
        reduced.to_pickle(output/f'features_{months}m.pkl.gz',compression='gzip')
        write_json(output/f'feature_dictionary_{months}m.json',{c:dictionaries[months][c] for c in reduced})
        audit=json.loads((source/f'audit_{months}m.json').read_text())
        diagnostics={}
        for column in removals[months]:
            s=X[column]
            diagnostics[column]=dict(windows=len(s),complete=int(np.isclose(s,1,atol=1e-6,rtol=0).sum()),
                zero=int(s.eq(0).sum()),missing=int(s.isna().sum()),mean_coverage_fraction=float(s.mean()))
        audit.update(feature_count=len(reduced.columns),all_missing_features=int(reduced.isna().all().sum()),
            coverage_fractions_in_predictors=False,seizure_coverage_fraction_diagnostics=diagnostics,
            removed_predictors=removals[months],source_feature_manifest_sha256=source_hash)
        write_json(output/f'audit_{months}m.json',audit)
    new_provenance=copy.deepcopy(provenance)
    new_provenance.update(feature_schema=HISTORICAL_SCHEMA,
        prepared_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        preparation='Derive from checksummed saved features; remove coverage fractions only; do not recompute labels or other features',
        source_feature_directory=str(source),source_feature_manifest_sha256=source_hash,
        removed_predictors={f'{m}m':columns for m,columns in removals.items()},
        source_build_code_sha256=provenance['code_sha256'],
        code_sha256={p.relative_to(ROOT.parent).as_posix():digest(p) for p in (Path(__file__),ROOT/'feature_schema.py')})
    write_json(output/'provenance.json',new_provenance)
    new_manifest=copy.deepcopy(manifest)
    new_manifest['feature_schema']=HISTORICAL_SCHEMA
    for months,X in frames.items():
        new_manifest['windows'][f'{months}m']['features']=len(X.columns)-len(removals[months])
    new_manifest['files']={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    write_json(output/'manifest.json',new_manifest)
    # Reload through the production checksum/alignment validator, then verify derivation.
    derived,new_visits,new_targets,_,new_split,_,_=load_status_bundle(output)
    pd.testing.assert_frame_equal(visits,new_visits)
    pd.testing.assert_frame_equal(targets,new_targets)
    if new_split!=split:
        raise AssertionError('Patient assignments changed')
    for months,X in derived.items():
        pd.testing.assert_frame_equal(frames[months].drop(columns=removals[months]),X)
    if digest(source/'manifest.json')!=source_hash:
        raise AssertionError('Source bundle changed during derivation')
    return new_manifest


METRICS = ('roc_auc', 'average_precision', 'brier', 'log_loss', 'accuracy_at_0_5',
    'sensitivity_at_0_5', 'specificity_at_0_5', 'precision_at_0_5', 'f1_at_0_5', 'npv_at_0_5')


def safe_ratio(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=b != 0)


def cluster_bootstrap(y, p, ids, replicates=1000, seed=42):
    y, p, ids = np.asarray(y, int), np.asarray(p, float), np.asarray(ids)
    codes, patients = pd.factorize(ids, sort=True)
    n = len(patients)
    multiplicity = np.random.default_rng(seed).multinomial(n, np.full(n, 1 / n), size=replicates)
    w = patient_weights(ids)
    predicted = p >= .5
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    values = {}
    for name, v in [('brier', (p-y)**2), ('log_loss', loss), ('accuracy_at_0_5', predicted == y),
        ('tn', (y == 0) & ~predicted), ('fp', (y == 0) & predicted),
        ('fn', (y == 1) & ~predicted), ('tp', (y == 1) & predicted)]:
        per_patient = np.bincount(codes, weights=np.asarray(v)*w, minlength=n)
        values[name] = multiplicity @ per_patient / n
    for name, a, b in [('sensitivity_at_0_5','tp','fn'), ('specificity_at_0_5','tn','fp'),
        ('precision_at_0_5','tp','fp'), ('npv_at_0_5','tn','fn')]:
        values[name] = safe_ratio(values[a], values[a] + values[b])
    values['f1_at_0_5'] = safe_ratio(2 * values['tp'], 2 * values['tp'] + values['fp'] + values['fn'])
    order = np.argsort(-p, kind='stable')
    boundaries = np.r_[np.flatnonzero(np.diff(p[order]) != 0), len(y)-1]
    auc, ap = [], []
    for start in range(0, replicates, 32):
        weights = multiplicity[start:start+32, codes[order]] * w[order]
        tp = np.cumsum(weights*y[order], axis=1)[:,boundaries]
        fp = np.cumsum(weights*(1-y[order]), axis=1)[:,boundaries]
        tpr, fpr = safe_ratio(tp,tp[:,-1:]), safe_ratio(fp,fp[:,-1:])
        dtpr = np.diff(np.column_stack([np.zeros(len(tp)),tpr]), axis=1)
        dfpr = np.diff(np.column_stack([np.zeros(len(tp)),fpr]), axis=1)
        prior = np.column_stack([np.zeros(len(tp)),tpr[:,:-1]])
        auc.extend(np.sum(dfpr*(tpr+prior)/2, axis=1))
        average_precision = np.sum(dtpr*np.nan_to_num(safe_ratio(tp,tp+fp)), axis=1)
        average_precision[(tp[:,-1] == 0) | (fp[:,-1] == 0)] = np.nan
        ap.extend(average_precision)
    values['roc_auc'],values['average_precision'] = np.asarray(auc),np.asarray(ap)
    return {name: values[name] for name in METRICS}


def interval(values):
    valid = np.asarray(values)[np.isfinite(values)]
    return np.quantile(valid,[.025,.975]).tolist() if len(valid)>=2 else None


def paired_summary(y, p, ids, baseline, replicates=1000, seed=42):
    result = performance(y,p,ids,bootstrap=0,seed=seed)
    before = performance(y,baseline,ids,bootstrap=0,seed=seed)
    boot = cluster_bootstrap(y,p,ids,replicates,seed)
    old_boot = cluster_bootstrap(y,baseline,ids,replicates,seed)
    result['bootstrap_95'] = {name:interval(boot[name]) for name in METRICS}
    result['paired_difference_from_baseline'] = {name:dict(
        estimate=result[name]-before[name] if result[name] is not None and before[name] is not None else None,
        bootstrap_95=interval(boot[name]-old_boot[name]),
        defined_replicates=int(np.isfinite(boot[name]-old_boot[name]).sum())) for name in METRICS}
    return result


def file_hashes(directory):
    return {p.relative_to(directory).as_posix():digest(p) for p in sorted(directory.rglob('*'))
        if p.is_file() and p.name != 'artifact_manifest.json'}


def fmt(value):
    return 'undefined' if value is None else f'{value:.4f}'


def verify_actual_visits(args):
    """Reproduce the historical comparison of actual-visit medication features."""
    old_root, new_root = args.baseline_results.resolve(), args.results.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new verification directory')
    output.mkdir(parents=True, exist_ok=True)
    old_hashes = file_hashes(old_root)
    new_prov = json.loads((new_root/'provenance.json').read_text())
    old_prov = json.loads((old_root/'provenance.json').read_text())
    new_dir, old_dir = Path(new_prov['feature_directory']), Path(old_prov['feature_directory'])
    before, old_visits, old_targets, _, old_split, _, _ = load_status_bundle(old_dir)
    after, visits, targets, _, split, _, feature_prov = load_status_bundle(new_dir)
    pd.testing.assert_frame_equal(visits,old_visits)
    pd.testing.assert_frame_equal(targets,old_targets)
    assert split == old_split
    original_manifest_hash = digest(old_dir/'manifest.json')
    snapshot = read_snapshot(feature_prov['snapshot'])
    raw_records = rows(snapshot.tables,'AsmRecords')
    clinic = {}
    for record in raw_records:
        if actual_clinical_visit(record.get('IsWuVisit')) and day(record.get('Date')):
            clinic.setdefault(str(record['PatientBaselineViewModelPatientId']),[]).append(record)
    for records in clinic.values():
        records.sort(key=lambda r:day(r['Date']))
    verified_pairs = {}
    evidence_blocks = []
    migration = []
    for months in (3,6,9):
        validate_active_features(after[months],months)
        for name in before[months]:
            action = 'removed' if name not in after[months] else 'redefined' if medication_comparison_feature(name) or name.startswith('asm_visits_') else 'unchanged'
            migration.append(dict(history_months=months,feature=name,action=action))
            if action == 'unchanged':
                pd.testing.assert_series_equal(before[months][name],after[months][name])
        audit = json.loads((new_dir/f'audit_{months}m.json').read_text())
        assert audit['medication_comparisons_enabled'] == (months != 3)
        if months == 3:
            assert not any(medication_comparison_feature(c) for c in after[months])
            continue
        evidence = audit['medication_comparison_evidence']
        assert len(evidence) == len(visits)
        paired = 0
        for index,e in enumerate(evidence):
            assert str(visits.iloc[index].patient_id)==e['patient_id']
            assert day(visits.iloc[index].prediction_date) == day(e['prediction_date'])
            start,end=day(e['input_start']),day(e['input_end_exclusive'])
            candidates=[r for r in clinic.get(e['patient_id'],[]) if start<=day(r['Date'])<end]
            assert len(candidates)==e['actual_visits']
            if len(candidates)>=2:
                old,new=candidates[-2:]
                assert e['previous_record_id']==str(old['Id']) and e['latest_record_id']==str(new['Id'])
                assert start<=day(old['Date'])<day(new['Date'])<end
                paired+=1
            else:
                assert e['previous_record_id'] is None and e['latest_record_id'] is None
                assert pd.isna(after[months].iloc[index].asm_dose_comparable_drugs)
        verified_pairs[f'{months}m']=dict(prediction_rows=len(evidence),actual_pairs=paired,
            assessable_pairs=sum(e['pair_assessable'] for e in evidence))
        block=pd.DataFrame(evidence)
        block['history_months']=months
        evidence_blocks.append(block)
    pd.DataFrame(migration).to_csv(output/'feature_changes.csv',index=False)
    pd.concat(evidence_blocks,ignore_index=True).to_csv(output/'comparison_visit_evidence.csv.gz',index=False,compression='gzip')
    known = targets.status_at_visit.notna() & visits['split'].eq('test')
    latest = latest_visit_indices(visits)
    supplied,label_audit,reference = apply_folder_labels(targets.loc[latest],args.label_root,args.label_overrides)
    latest = supplied.index[supplied.patient_id.isin(split['test']) & supplied.status_at_visit.notna()]
    label_audit.to_json(output/'latest_label_audit.jsonl',orient='records',lines=True)
    cohorts = dict(heldout_computed=(visits.index[known],targets.loc[known,'status_at_visit'].to_numpy(int)),
        latest_supplied=(latest,supplied.loc[latest,'status_at_visit'].to_numpy(int)))
    registries={name:json.loads((root/'models.json').read_text()) for name,root in [('baseline',old_root),('actual_visits',new_root)]}
    metrics,summary,predictions = {},[],[]
    for months in (3,6,9):
        metrics[f'{months}m']={}
        for scope,(indices,y) in cohorts.items():
            ids=visits.loc[indices,'patient_id']
            metrics[f'{months}m'][scope]={}
            for variant in VARIANTS:
                p={}
                for name,root,data in [('baseline',old_root,before),('actual_visits',new_root,after)]:
                    model=joblib.load(root/registries[name][f'visit_status_history_{months}m_{variant}'])
                    p[name]=model.predict_dre_probability(data[months].loc[indices])
                    b=visits.loc[indices].copy()
                    b['row_index']=indices
                    b['history_months'],b['scope'],b['configuration'],b['variant']=months,scope,name,variant
                    b['observed_status'],b['probability']=y,p[name]
                    predictions.append(b)
                metrics[f'{months}m'][scope][variant]={}
                for name in p:
                    m=paired_summary(y,p[name],ids,p['baseline'])
                    metrics[f'{months}m'][scope][variant][name]=m
                    summary.append(dict(history_months=months,scope=scope,model=variant,configuration=name,
                        **{key:m[key] for key in METRICS}))
    write_json(output/'metrics.json',metrics)
    pd.DataFrame(summary).to_csv(output/'metrics_summary.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/'predictions.csv.gz',index=False,compression='gzip')
    verification=dict(unchanged_prediction_rows=True,unchanged_outcomes=True,unchanged_patient_split=True,
        unchanged_noncomparison_features=True,verified_actual_pairs=verified_pairs,clinical_visit_policy=CLINICAL_VISIT_POLICY,
        source_legend='docx_parsers/docx_parser_asm.py: original header says not Wu:0; Wu:1; no visit record:-',
        three_month_medication_comparisons_removed=True,
        limitations=['Prediction rows and outcome calculations remain the original dated medication assessments. This change corrects predictor comparisons only.',
            'Latest medication levels still use the latest available medication record; comparison endpoints use actual visits.',
            'This is the same held-out cohort previously evaluated; no independent external validation.'])
    assert file_hashes(old_root)==old_hashes
    assert digest(old_dir/'manifest.json')==original_manifest_hash
    write_json(output/'verification.json',verification)
    write_json(output/'provenance.json',dict(results=str(new_root),baseline=str(old_root),feature_directory=str(new_dir),
        feature_manifest_sha256=digest(new_dir/'manifest.json'),baseline_files_sha256=old_hashes,
        snapshot_manifest_sha256=feature_prov['snapshot_manifest_sha256'],threshold=.5,patient_weighting='Equal total weight per patient in each cohort',
        bootstrap_replicates=1000,forest_seed=42,latest_reference=reference,
        code_sha256={name:digest(ROOT/name) for name in ('historical_audit.py','feature_schema.py','longitudinal_features.py')}))
    lines=['# Medication comparisons at actual clinical visits','',
        'The 3-month model excludes the dose-comparison count and all medication-comparison predictors. '
        'For 6 and 9 months, dose differences, counts of added/stopped medicines, regimen changes, '
        'composition-change histories and visit-gap features use actual distinct visits inside the input window.','',
        CLINICAL_VISIT_POLICY,'',
        'Identical prescriptions at separate confirmed visits remain valid comparisons. Monthly nonvisit rows '
        'do not become additional visits. Missing markers are excluded; missing medication data at a real visit '
        'are not bypassed by choosing an older visit. Fewer than two eligible visits leave changes missing.','',
        '## Verification','',
        'All 43,581 prediction rows, computed labels and patient assignments match the prior simplified bundle. '
        'Noncomparison features are unchanged except ASM visit counts, which now count actual clinic visits. '
        'The 12-month history anchor and target seizure interval remain. Only 3-, 6- and 9-month models are fitted.','',
        'Latest medication levels still use the latest available medication record. Prediction rows remain '
        'the existing dated medication assessments; the change does not redefine the target cohort as clinic-only.','',
        '## Model performance','',
        'All-feature and freshly selected top-20 forests retain 500 trees, depth 8, minimum leaf 10 and sqrt '
        'feature sampling, seed 42. Filtering, imputation and selection use training patients only. '
        'Computed historical visit labels are primary; supplied latest-visit labels are secondary. '
        'Folder labels never enter training. The same held-out patients were previously evaluated.','',
        'Metrics use threshold 0.5 and equal patient weighting. NPV is undefined when there are no negative predictions. '
        'metrics.json contains 1,000 paired patient-cluster bootstrap intervals for all metrics and differences. '
        'Positive Brier/log-loss differences indicate worsening; nonsignificant results do not establish equivalence.','',
        '| History | Model | Baseline AUROC | New AUROC | Baseline Brier | New Brier | Baseline NPV | New NPV | Baseline specificity | New specificity | Baseline F1 | New F1 | Brier difference [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for months in (3,6,9):
        for variant in VARIANTS:
            pair=metrics[f'{months}m']['heldout_computed'][variant]
            old,new=pair['baseline'],pair['actual_visits']
            d=new['paired_difference_from_baseline']['brier'];ci=d['bootstrap_95']
            lines.append(f'| {months}m | {variant} | {fmt(old["roc_auc"])} | {fmt(new["roc_auc"])} | '
                f'{fmt(old["brier"])} | {fmt(new["brier"])} | {fmt(old["npv_at_0_5"])} | {fmt(new["npv_at_0_5"])} | '
                f'{fmt(old["specificity_at_0_5"])} | {fmt(new["specificity_at_0_5"])} | {fmt(old["f1_at_0_5"])} | {fmt(new["f1_at_0_5"])} | '
                f'{d["estimate"]:+.4f} [{ci[0]:.4f}, {ci[1]:.4f}] |')
    lines+=['','## All metrics, including secondary latest labels','',
        '| History | Scope | Model | Configuration | '+' | '.join(METRICS)+' |',
        '|---|---|---|---|'+'|'.join(['---:']*len(METRICS))+'|']
    for r in summary:
        lines.append('| '+ ' | '.join([str(r['history_months']),r['scope'],r['model'],r['configuration']]+[fmt(r[k]) for k in METRICS])+' |')
    lines+=['','## Artifacts','',
        '- [Comparison visit IDs, dates and availability](comparison_visit_evidence.csv.gz).',
        '- [Feature changes](feature_changes.csv); [metrics](metrics.json); [predictions](predictions.csv.gz).',
        '- [Verification](verification.json); [provenance](provenance.json).',
        f'- Saved production models, selected features and plots: `{new_root}`.','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    for root in (new_root,output):
        write_json(root/'artifact_manifest.json',file_hashes(root))
    print('Verified and saved '+str(output/'report.md'),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    coverage = commands.add_parser('without-coverage', help='Rebuild a historical coverage-free feature bundle')
    coverage.add_argument('--feature-dir', type=Path, required=True)
    coverage.add_argument('--output-dir', type=Path, required=True)
    comparison = commands.add_parser('actual-visits', help='Reproduce the historical medication-visit comparison')
    comparison.add_argument('--baseline-results', type=Path, default=ROOT / 'results/rf_visit_status_20261025_simplified')
    comparison.add_argument('--results', type=Path, default=ROOT / 'results/rf_visit_status_20261025_actual_visits')
    comparison.add_argument('--output', type=Path, default=ROOT / 'results/actual_visit_comparison_20261025')
    comparison.add_argument('--label-root', type=Path, default=ROOT.parent)
    comparison.add_argument('--label-overrides', type=Path, default=ROOT / 'latest_status_overrides.json')
    args = parser.parse_args()
    if args.command == 'actual-visits':
        verify_actual_visits(args)
    else:
        manifest = prepare_without_coverage(args.feature_dir, args.output_dir)
        counts = {window: metadata['features'] for window, metadata in manifest['windows'].items()}
        print(f'Saved coverage-free feature bundle: {args.output_dir.resolve()}', flush=True)
        print(f'Feature counts: {counts}', flush=True)


if __name__ == '__main__':
    main()


