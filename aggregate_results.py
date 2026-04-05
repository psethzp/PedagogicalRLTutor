#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from transformers import AutoTokenizer

INTERNAL_PATTERNS = {
    'delta_solve_rate': r'Delta mean:\s*([-+eE0-9.]+)',
    'leak_solution': r'Leaked solutions mean:\s*([-+eE0-9.]+)',
    'ped_rm_macro': r'Pedagogical reward mean macro avg:\s*([-+eE0-9.]+)',
    'ped_rm_micro': r'Pedagogical reward mean micro avg:\s*([-+eE0-9.]+)',
}

TASK_TO_COLUMN = {
    'problem_solving': 'problem_solving',
    'socratic_questioning': 'socratic_questioning',
    'student_solution_correctness': 'student_solution_correctness',
    'mistake_location': 'mistake_location',
    'mistake_correction': 'mistake_correction',
    'scaffolding_generation': 'scaffolding_generation',
    'pedagogy_following': 'pedagogy_following',
    'scaffolding_generation_hard': 'scaffolding_generation_hard',
    'pedagogy_following_hard': 'pedagogy_following_hard',
}

MODELS = [
    {
        'label': 'TutorRL-7B',
        'internal_log': 'PedagogicalRL/logs/internal_tutorrly7b.log',
        'internal_metrics_json': 'PedagogicalRL/eval_outputs/tutorrly7b_metrics.json',
        'external_yaml': 'mathtutorbench/results/results-TutorRL-7B.yaml',
        'benchmark_meta': 'mathtutorbench/results/benchmark_meta-TutorRL-7B.json',
        'time_log': None,
        'gpu_csv': None,
        'gen_prefix': 'TutorRL-7B',
    },
    {
        'label': 'TutorRL-7B-think',
        'internal_log': 'PedagogicalRL/logs/internal_tutorrly7b_think.log',
        'internal_metrics_json': 'PedagogicalRL/eval_outputs/tutorrly7b_think_metrics.json',
        'external_yaml': 'mathtutorbench/results/results-TutorRL-7B-think.yaml',
        'benchmark_meta': 'mathtutorbench/results/benchmark_meta-TutorRL-7B-think.json',
        'time_log': None,
        'gpu_csv': None,
        'gen_prefix': 'TutorRL-7B-think',
    },
    {
        'label': 'TutorRM+GRPO',
        'internal_log': 'PedagogicalRL/logs/internal_tutorrm_grpo.log',
        'internal_metrics_json': 'PedagogicalRL/outputs/tutorrm_grpo/eval_outputs/metrics.json',
        'external_yaml': 'mathtutorbench/results/results-tutorrm-grpo.yaml',
        'benchmark_meta': 'mathtutorbench/results/benchmark_meta-tutorrm-grpo.json',
        'time_log': 'PedagogicalRL/logs/final_grpo.log',
        'gpu_csv': 'PedagogicalRL/logs/gpu0.csv',
        'gen_prefix': 'tutorrm-grpo',
    },
    {
        'label': 'TutorRM+GDPO',
        'internal_log': 'PedagogicalRL/logs/internal_tutorrm_gdpo.log',
        'internal_metrics_json': 'PedagogicalRL/outputs/tutorrm_gdpo/eval_outputs/metrics.json',
        'external_yaml': 'mathtutorbench/results/results-tutorrm-gdpo.yaml',
        'benchmark_meta': 'mathtutorbench/results/benchmark_meta-tutorrm-gdpo.json',
        'time_log': 'PedagogicalRL/logs/final_gdpo.log',
        'gpu_csv': 'PedagogicalRL/logs/gpu1.csv',
        'gen_prefix': 'tutorrm-gdpo',
    },
]

PREFERRED_KEYS = ['win_rate', 'accuracy', 'acc', 'score', 'bleu', 'sacrebleu', 'f1', 'exact_match']
GENERATION_TEXT_KEYS = ['generated_teacher_utterance', 'prediction', 'generated_text', 'response']


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8', errors='ignore') if path.exists() else ''


def first_numeric_leaf(obj: Any):
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return float(obj)
    if isinstance(obj, dict):
        for key in PREFERRED_KEYS:
            if key in obj and isinstance(obj[key], (int, float)):
                return float(obj[key])
        if 'results' in obj:
            found = first_numeric_leaf(obj['results'])
            if found is not None:
                return found
        for value in obj.values():
            found = first_numeric_leaf(value)
            if found is not None:
                return found
    if isinstance(obj, list):
        for value in obj:
            found = first_numeric_leaf(value)
            if found is not None:
                return found
    return None


def parse_internal_metrics_json(path: Path):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return {
        'delta_solve_rate': data.get('delta_mean'),
        'leak_solution': data.get('leaked_solutions_mean'),
        'ped_rm_micro': data.get('pedagogical_reward_micro_avg'),
        'ped_rm_macro': data.get('pedagogical_reward_macro_avg'),
    }


def parse_internal_metrics_log(path: Path):
    text = read_text(path)
    out = {k: None for k in INTERNAL_PATTERNS}
    for key, pat in INTERNAL_PATTERNS.items():
        m = re.search(pat, text)
        if m:
            out[key] = float(m.group(1))
    return out


def parse_external_metrics(yaml_path: Path):
    if not yaml_path.exists():
        return {v: None for v in TASK_TO_COLUMN.values()}
    data = yaml.safe_load(yaml_path.read_text(encoding='utf-8')) or {}
    out = {v: None for v in TASK_TO_COLUMN.values()}
    for task_name, col in TASK_TO_COLUMN.items():
        if task_name in data:
            out[col] = first_numeric_leaf(data[task_name])
    return out


def parse_time_hours(log_path: Optional[Path]):
    if log_path is None or not log_path.exists():
        return None
    text = read_text(log_path)
    m = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:]+(?:\.[0-9]+)?)', text)
    if not m:
        return None
    s = m.group(1)
    parts = s.split(':')
    if len(parts) == 3:
        h, m_, s_ = parts
    elif len(parts) == 2:
        h = '0'
        m_, s_ = parts
    else:
        return None
    return int(h) + int(m_) / 60.0 + float(s_) / 3600.0


def parse_peak_gpu_mem_gb(csv_path: Optional[Path]):
    if csv_path is None or not csv_path.exists():
        return None
    peak = 0.0
    with csv_path.open() as f:
        for line in f:
            if 'memory.used' in line.lower():
                continue
            parts = [p.strip() for p in line.split(',')]
            if len(parts) < 4:
                continue
            mem = parts[3].split()[0]
            try:
                peak = max(peak, float(mem))
            except ValueError:
                pass
    return peak / 1024.0 if peak else None


def parse_benchmark_minutes(meta_path: Optional[Path]):
    if meta_path is None or not meta_path.exists():
        return None
    data = json.loads(meta_path.read_text())
    return data.get('elapsed_minutes')


def compute_avg_tokens(project_root: Path, model_prefix: str, tokenizer_name: str):
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    task_files = list((project_root / 'mathtutorbench' / 'results').glob(f'generations-{model_prefix}-*.json'))
    if not task_files:
        return None
    total = 0
    n = 0
    for path in task_files:
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        for item in data:
            text = ''
            for key in GENERATION_TEXT_KEYS:
                if key in item and isinstance(item[key], str):
                    text = item[key]
                    break
            if not text:
                continue
            total += len(tok.encode(text))
            n += 1
    return total / n if n else None


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', required=True)
    ap.add_argument('--output-dir', default=None)
    ap.add_argument('--tokenizer-name', default='Qwen/Qwen2.5-7B-Instruct')
    args = ap.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else project_root / 'summary_tables'

    internal_rows = []
    external_rows = []
    efficiency_rows = []

    for spec in MODELS:
        internal = parse_internal_metrics_json(project_root / spec['internal_metrics_json'])
        if internal is None:
            internal = parse_internal_metrics_log(project_root / spec['internal_log'])
        external = parse_external_metrics(project_root / spec['external_yaml'])
        internal_rows.append({'model': spec['label'], **internal})
        external_rows.append({'model': spec['label'], **external})

        if spec['time_log'] is not None:
            efficiency_rows.append({
                'model': spec['label'],
                'train_hours': parse_time_hours(project_root / spec['time_log']),
                'peak_gpu_mem_gb': parse_peak_gpu_mem_gb(project_root / spec['gpu_csv']) if spec['gpu_csv'] else None,
                'mathtutorbench_minutes': parse_benchmark_minutes(project_root / spec['benchmark_meta']),
                'avg_tutor_response_tokens': compute_avg_tokens(project_root, spec['gen_prefix'], args.tokenizer_name),
            })

    write_csv(out_dir / 'results_internal.csv', internal_rows, ['model', 'delta_solve_rate', 'leak_solution', 'ped_rm_micro', 'ped_rm_macro'])
    write_csv(out_dir / 'results_external.csv', external_rows, ['model', 'problem_solving', 'socratic_questioning', 'student_solution_correctness', 'mistake_location', 'mistake_correction', 'scaffolding_generation', 'pedagogy_following', 'scaffolding_generation_hard', 'pedagogy_following_hard'])
    write_csv(out_dir / 'results_efficiency.csv', efficiency_rows, ['model', 'train_hours', 'peak_gpu_mem_gb', 'mathtutorbench_minutes', 'avg_tutor_response_tokens'])

    print(f'Wrote summary tables to {out_dir}')


if __name__ == '__main__':
    main()
