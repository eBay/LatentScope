import os
import json
from loguru import logger
from utils import  metric_type_map
import yaml
import random
from models import CaseMetric
from circa.graph.structural import StructuralGraph
from typing import *


class StructuralGraphBuilder:
    def __init__(self, dataset: str):
        self.dataset = dataset
        # Load deps
        logger.info(f"[CircaGraphBuilder] Loading deps for {dataset}...")

        with open(os.path.join('data', dataset, 'labels', 'service_deps.json'), 'rt') as f:
            self.deps = json.load(f)

    def build(self, metrics: CaseMetric):
        # Build links with circa
        s_map = {}
        metric_data = {}
        for key, entity, value in zip(metrics.key, metrics.entity, metrics.value):
            # Save into s_map
            s_map.setdefault(entity, {
                'name': entity,
                'mappings': {}
            })

            cur_type = metric_type_map(self.dataset, key)

            s_map[entity]['mappings'][key] = [{
                'type': cur_type,
                'component': entity
            }]

            metric_data.setdefault(entity, {})
            metric_data[entity][key] = list(zip(metrics.ts, value))

        # Add Deps
        for (u, v) in self.deps:
            if (u in s_map and v in s_map and u != v):
                s_map[u].setdefault('dependencies', {u: []})
                s_map[u]['dependencies'][u].append(v)

        # Build structural graph with circa tool
        # Dump s_map into yml
        sgraph_basic = yaml.safe_load(open(f'data/{self.dataset}/labels/metagraph_template.yml', 'rt'))
        sgraph_basic['components'] = list(s_map.values())

        os.makedirs(f'data/{self.dataset}/sgraph', exist_ok=True)
        filename = f'data/{self.dataset}/sgraph/{random.randint(0, 1e9)}.yml'
        with open(filename, 'wt') as f:
            yaml.safe_dump(sgraph_basic, f)
        
        # Build graph
        sgraph = StructuralGraph(filename=filename)
        metric_graph = sgraph.visit(
            {
                entity: set(metric_names)
                for entity, metric_names in metric_data.items()
            }
        )
        os.remove(filename)

        # Map from metric to idx
        metric_idx_dict = dict(zip(metrics.key, range(len(metrics.key))))

        # Fetch information from sgraph
        graph_edges = []
        for (u, v) in metric_graph.edges:
            graph_edges.append((metric_idx_dict[u.metric], metric_idx_dict[v.metric]))

        return graph_edges
