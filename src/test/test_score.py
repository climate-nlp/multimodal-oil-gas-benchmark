import unittest
import os
import shlex
import subprocess
import json
from typing import Dict, List, Tuple
import tempfile
import src.score as scorer


def run_eval_proc(s: List[Dict], g: List[Dict]) -> List[Tuple]:
    with tempfile.TemporaryDirectory() as d:
        fs = os.path.join(d, 's')
        fg = os.path.join(d, 'g')
        with open(fs, 'w') as f:
            for l in s:
                f.write(json.dumps(l) + '\n')
        with open(fg, 'w') as f:
            for l in g:
                f.write(json.dumps(l) + '\n')
        subproc = subprocess.run(
            shlex.split(f'python {scorer.__file__} -s {fs} -g {fg}'),
            encoding='utf-8',
            stdout=subprocess.PIPE
        )
    scores = json.loads(subproc.stdout)
    # Create tuples of (label, metric, value)
    scores = [(d['label'], d['metric'], d['value']) for d in scores]
    return scores


class TestScorer(unittest.TestCase):

    def test1(self):
        # Empty
        system = [
            {
                'id': 'video001',
                'labels': [],
            },
        ]
        gold = system
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 0), res)
        self.assertIn(('All', 'n_system', 0), res)
        self.assertIn(('All', 'n_correct', 0), res)
        self.assertIn(('All', 'precision', 0), res)
        self.assertIn(('All', 'recall', 0), res)
        self.assertIn(('All', 'f', 0), res)

    def test2(self):
        # Exact match
        system = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA'],
            },
        ]
        gold = system
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 2), res)
        self.assertIn(('All', 'n_system', 2), res)
        self.assertIn(('All', 'n_correct', 2), res)
        self.assertIn(('All', 'precision', 100), res)
        self.assertIn(('All', 'recall', 100), res)
        self.assertIn(('All', 'f', 100), res)

        self.assertIn(('CA', 'n_gold', 1), res)
        self.assertIn(('CA', 'n_system', 1), res)
        self.assertIn(('CA', 'n_correct', 1), res)
        self.assertIn(('CA', 'precision', 100), res)
        self.assertIn(('CA', 'recall', 100), res)
        self.assertIn(('CA', 'f', 100), res)

        self.assertIn(('PA', 'n_gold', 1), res)
        self.assertIn(('PA', 'n_system', 1), res)
        self.assertIn(('PA', 'n_correct', 1), res)
        self.assertIn(('PA', 'precision', 100), res)
        self.assertIn(('PA', 'recall', 100), res)
        self.assertIn(('PA', 'f', 100), res)

    def test3(self):
        # Exact match but invalid input
        system = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA', 'PA', 'CA'],
            },
        ]
        gold = system
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 2), res)
        self.assertIn(('All', 'n_system', 2), res)
        self.assertIn(('All', 'n_correct', 2), res)
        self.assertIn(('All', 'precision', 100), res)
        self.assertIn(('All', 'recall', 100), res)
        self.assertIn(('All', 'f', 100), res)

        self.assertIn(('CA', 'n_gold', 1), res)
        self.assertIn(('CA', 'n_system', 1), res)
        self.assertIn(('CA', 'n_correct', 1), res)
        self.assertIn(('CA', 'precision', 100), res)
        self.assertIn(('CA', 'recall', 100), res)
        self.assertIn(('CA', 'f', 100), res)

        self.assertIn(('PA', 'n_gold', 1), res)
        self.assertIn(('PA', 'n_system', 1), res)
        self.assertIn(('PA', 'n_correct', 1), res)
        self.assertIn(('PA', 'precision', 100), res)
        self.assertIn(('PA', 'recall', 100), res)
        self.assertIn(('PA', 'f', 100), res)

    def test4(self):
        # Perfect recall
        system = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA', 'SA'],
            },
        ]
        gold = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA'],
            },
        ]
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 2), res)
        self.assertIn(('All', 'n_system', 3), res)
        self.assertIn(('All', 'n_correct', 2), res)
        self.assertIn(('All', 'precision', 66.6), res)
        self.assertIn(('All', 'recall', 100), res)
        self.assertIn(('All', 'f', 80.0), res)

        self.assertIn(('CA', 'n_gold', 1), res)
        self.assertIn(('CA', 'n_system', 1), res)
        self.assertIn(('CA', 'n_correct', 1), res)
        self.assertIn(('CA', 'precision', 100), res)
        self.assertIn(('CA', 'recall', 100), res)
        self.assertIn(('CA', 'f', 100), res)

        self.assertIn(('PA', 'n_gold', 1), res)
        self.assertIn(('PA', 'n_system', 1), res)
        self.assertIn(('PA', 'n_correct', 1), res)
        self.assertIn(('PA', 'precision', 100), res)
        self.assertIn(('PA', 'recall', 100), res)
        self.assertIn(('PA', 'f', 100), res)

    def test5(self):
        # Perfect precision
        system = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA'],
            },
        ]
        gold = [
            {
                'id': 'video001',
                'labels': ['CA', 'PA', 'SA'],
            },
        ]
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 3), res)
        self.assertIn(('All', 'n_system', 2), res)
        self.assertIn(('All', 'n_correct', 2), res)
        self.assertIn(('All', 'precision', 100), res)
        self.assertIn(('All', 'recall', 66.6), res)
        self.assertIn(('All', 'f', 80.0), res)

        self.assertIn(('CA', 'n_gold', 1), res)
        self.assertIn(('CA', 'n_system', 1), res)
        self.assertIn(('CA', 'n_correct', 1), res)
        self.assertIn(('CA', 'precision', 100), res)
        self.assertIn(('CA', 'recall', 100), res)
        self.assertIn(('CA', 'f', 100), res)

        self.assertIn(('PA', 'n_gold', 1), res)
        self.assertIn(('PA', 'n_system', 1), res)
        self.assertIn(('PA', 'n_correct', 1), res)
        self.assertIn(('PA', 'precision', 100), res)
        self.assertIn(('PA', 'recall', 100), res)
        self.assertIn(('PA', 'f', 100), res)

        self.assertIn(('SA', 'n_gold', 1), res)
        self.assertIn(('SA', 'n_system', 0), res)
        self.assertIn(('SA', 'n_correct', 0), res)
        self.assertIn(('SA', 'precision', 0), res)
        self.assertIn(('SA', 'recall', 0), res)
        self.assertIn(('SA', 'f', 0), res)

    def test6(self):
        # Multiple videos and different order
        system = [
            {
                'id': 'video001',
                'labels': ["Economy and Business", "Work", "Environment", "Community and Life"],
            },
            {
                'id': 'video002',
                'labels': ["Economy and Business"]
            },
        ]
        gold = [
            {
                'id': 'video002',
                'labels': []
            },
            {
                'id': 'video001',
                'labels': ["Economy and Business", "Community and Life", "Patriotism"],
            },
        ]
        res = run_eval_proc(system, gold)
        self.assertIn(('All', 'n_gold', 3), res)
        self.assertIn(('All', 'n_system', 5), res)
        self.assertIn(('All', 'n_correct', 2), res)
        self.assertIn(('All', 'precision', 40.0), res)
        self.assertIn(('All', 'recall', 66.6), res)
        self.assertIn(('All', 'f', 50.0), res)

        self.assertIn(('Economy and Business', 'n_gold', 1), res)
        self.assertIn(('Economy and Business', 'n_system', 2), res)
        self.assertIn(('Economy and Business', 'n_correct', 1), res)
        self.assertIn(('Economy and Business', 'precision', 50.0), res)
        self.assertIn(('Economy and Business', 'recall', 100.0), res)
        # 100 * (2*(1/2)*(1))/((1/2)+(1))
        self.assertIn(('Economy and Business', 'f', 66.6), res)

        self.assertIn(('Community and Life', 'n_gold', 1), res)
        self.assertIn(('Community and Life', 'n_system', 1), res)
        self.assertIn(('Community and Life', 'n_correct', 1), res)
        self.assertIn(('Community and Life', 'precision', 100.0), res)
        self.assertIn(('Community and Life', 'recall', 100.0), res)
        self.assertIn(('Community and Life', 'f', 100.0), res)

        self.assertIn(('Patriotism', 'n_gold', 1), res)
        self.assertIn(('Patriotism', 'n_system', 0), res)
        self.assertIn(('Patriotism', 'n_correct', 0), res)
        self.assertIn(('Patriotism', 'precision', 0.0), res)
        self.assertIn(('Patriotism', 'recall', 0.0), res)
        self.assertIn(('Patriotism', 'f', 0.0), res)
