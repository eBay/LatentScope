import numpy as np
import json
import os
from loguru import logger
from models import CaseMetric, RootCause, RCC
import click
from typing import *


class Evaluator:
    def __init__(self, dataset: str, top_ks=[1, 5, 10], mrr_ks=[3, 5, 10]):
        self.dataset = dataset

        with open(os.path.join('data', dataset, 'labels', 'label.json'), 'rt') as f:
            self.label = json.load(f)

        self.top_ks = top_ks
        self.mrr_ks = mrr_ks
        self.result = {}
        self.case_type: Dict[str, str] = {}

    def evaluate_case(self, case: str, rank: List[RootCause], metrics: CaseMetric):
        self.result[case] = {}
        cur_rc = self.label[case]['rc']
        self.case_type[case] = cur_rc[0]['type']

        # Rank
        min_rank = 999999999
        min_ranks = [min_rank]
        for rc in cur_rc:
            if rc not in rank:
                continue
            
            rc_score = rank[rank.index(rc)].score
            candidate_ranks = [(i + 1) for i in range(len(rank)) if abs(rank[i].score - rc_score) < 1e-3]
            if candidate_ranks[0] < min_rank:
                min_rank = int(np.mean(candidate_ranks) + 0.5)
                min_ranks = candidate_ranks
        
        self.result[case]['topk'] = []
        for i, k in enumerate(self.top_ks):
            cur_scores = []
            for min_rank in min_ranks:
                if min_rank <= k:
                    cur_scores.append(1)
                else:
                    cur_scores.append(0)
            self.result[case]['topk'].append(np.mean(cur_scores))
        self.result[case]['rank'] = np.mean(min_ranks)


    def print_result(self):
        # Report the result
        report: str = "--------------Report-------------\n"
        for case in sorted(self.result):
            report += f"Case {case}: "
            for j, k in enumerate(self.top_ks):
                report += f"Top {k}: {self.result[case]['topk'][j]}, "
            report += f"Rank: {self.result[case]['rank']}, "
            report += f"Type: {self.case_type[case]}"
            report += '\n'

        # Calculate avg value
        valid_cases = [i for i in self.result.values() if 'rank' in i and 'topk' in i]
        topk = np.mean(np.stack([i['topk'] for i in valid_cases]), axis=0)
        report += f"============Micro ALL ({len(valid_cases)})==============\n"
        for j, k in enumerate(self.top_ks):
            report += f"Top {k}: {topk[j]:.4f}, "
        report += "\n"

        # Calculate MRR
        overall_mrr = np.mean(1 / np.array([i['rank'] for i in valid_cases]))
        report += f"MRR: {overall_mrr:.4f}"
        for k in self.mrr_ks:
            cur_mrr = float(np.mean([(1.0 / i['rank'] if i['rank'] < k else 0.0) for i in valid_cases]))
            report += f", MRR@{k}: {cur_mrr:.4f}"
        report += '\n'

        # Calculate Type Value
        per_type_values: Dict[str, List[float]] = {}
        for cur_type in set(self.case_type.values()):
            valid_cases = [self.result[case] for case in self.result if 'rank' in self.result[case] and self.case_type[case] == cur_type]
            topk = np.mean(np.stack([i['topk'] for i in valid_cases]), axis=0)
            report += f"============{cur_type} cases ({len(valid_cases)})==============\n"
            for j, k in enumerate(self.top_ks):
                per_type_values.setdefault(f"Top {k}", [])
                per_type_values[f"Top {k}"].append(topk[j])
                report += f"Top {k}: {topk[j]:.4f}, "
            report += "\n"
            mrr = np.mean(1 / np.array([i['rank'] for i in valid_cases]))
            report += f"MRR: {mrr:.4f}"
            per_type_values.setdefault(f"MRR", [])
            per_type_values[f"MRR"].append(mrr)
            for k in self.mrr_ks:
                cur_mrr = float(np.mean([(1.0 / i['rank'] if i['rank'] < k else 0.0) for i in valid_cases]))
                report += f", MRR@{k}: {cur_mrr:.4f}"
                per_type_values.setdefault(f"MRR@{k}", [])
                per_type_values[f"MRR@{k}"].append(cur_mrr)
            report += '\n'

        # Show Macro values
        report += f"==================Macro Values======================\n"
        for k in per_type_values:
            report += f"{k}: {np.mean(per_type_values[k])}\t"
        report += '\n'

        logger.info(report)
        return float(overall_mrr)

    def save_result(self, save_path: str):
        # Report the result
        report: str = "--------------Report-------------\n"
        for case in sorted(self.result):
            report += f"Case {case}: "
            for j, k in enumerate(self.top_ks):
                report += f"Top {k}: {self.result[case]['topk'][j]}, "
            report += f"Rank: {self.result[case]['rank']}, "
            report += f"Type: {self.case_type[case]}"
            report += '\n'

        # Calculate avg value
        valid_cases = [i for i in self.result.values() if 'rank' in i and 'topk' in i]
        topk = np.mean(np.stack([i['topk'] for i in valid_cases]), axis=0)
        report += f"============Micro ALL ({len(valid_cases)})==============\n"
        for j, k in enumerate(self.top_ks):
            report += f"Top {k}: {topk[j]:.4f}, "
        report += "\n"

        # Calculate MRR
        mrr = np.mean(1 / np.array([i['rank'] for i in valid_cases]))
        report += f"MRR: {mrr:.4f}"
        for k in self.mrr_ks:
            cur_mrr = float(np.mean([(1.0 / i['rank'] if i['rank'] < k else 0.0) for i in valid_cases]))
            report += f", MRR@{k}: {cur_mrr:.4f}"
        report += '\n'

        # Calculate Type Value
        per_type_values: Dict[str, List[float]] = {}
        for cur_type in set(self.case_type.values()):
            valid_cases = [self.result[case] for case in self.result if 'rank' in self.result[case] and self.case_type[case] == cur_type]
            topk = np.mean(np.stack([i['topk'] for i in valid_cases]), axis=0)
            report += f"============{cur_type} cases ({len(valid_cases)})==============\n"
            for j, k in enumerate(self.top_ks):
                per_type_values.setdefault(f"Top {k}", [])
                per_type_values[f"Top {k}"].append(topk[j])
                report += f"Top {k}: {topk[j]:.4f}, "
            report += "\n"
            mrr = np.mean(1 / np.array([i['rank'] for i in valid_cases]))
            report += f"MRR: {mrr:.4f}"
            per_type_values.setdefault(f"MRR", [])
            per_type_values[f"MRR"].append(mrr)
            for k in self.mrr_ks:
                cur_mrr = float(np.mean([(1.0 / i['rank'] if i['rank'] < k else 0.0) for i in valid_cases]))
                report += f", MRR@{k}: {cur_mrr:.4f}"
                per_type_values.setdefault(f"MRR@{k}", [])
                per_type_values[f"MRR@{k}"].append(cur_mrr)
            report += '\n'

        # Show Macro values
        report += f"==================Macro Values======================\n"
        for k in per_type_values:
            report += f"{k}: {np.mean(per_type_values[k])}\t"
        report += '\n'

        # Save result
        with open(save_path, 'wt') as f:
            f.write(report)
