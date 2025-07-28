# -*- coding: utf-8 -*-

BASE_REASON_PROMPT = """Please determine whether the input triple from a knowledge graph is correct or incorrect.
{test_triple}
Please return 'Y' if it is correct, otherwise return 'N'. Do not say anything else except your determination.
"""

RULE_PATH_PROMPT = """You are an expert in knowledge reasoning. Your task is to determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of reasoning paths from the knowledge graph.
The head entity is {head_entity}, the tail entity is {tail_entity}.
Reasoning rules inductively derived from the graph are:
{reasoning_rules}
Closed paths collected of the knowledge graph are:
{reasoning_paths}
The relation between the head and tail entity to be inferred is:
{test_triple}
Please return 'Y' if the triplet can be inferred from the knowledge graph based on the reasoning paths and rules provided, otherwise return 'N'. Do not say anything else except your determination.
"""

RULE_NE_PATH_PROMPT = """You are an expert in knowledge reasoning. Your task is to determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of reasoning paths from the knowledge graph.
The head entity is {head_entity}, the tail entity is {tail_entity}.
Neighbor triples from the knowledge graph are:
{neighbor_triples}
Reasoning rules inductively derived from the graph are:
{reasoning_rules}
Closed paths collected of the knowledge graph are:
{reasoning_paths}
The relation between the head and tail entity to be inferred is:
{test_triple}
Please return 'Y' if the triplet can be inferred from the knowledge graph based on the reasoning paths and rules provided, otherwise return 'N'. Do not say anything else except your determination.
"""

PATH_PROMPT = """You are an expert in knowledge reasoning. Your task is to determine whether the relation in the input can be reliably inferred between the head and tail entities, based on a set of reasoning paths from the knowledge graph.
The head entity is {head_entity}, the tail entity is {tail_entity}.
Closed paths collected of the knowledge graph are:
{reasoning_paths}
The relation between the head and tail entity to be inferred is:
{test_triple}
Please return 'Y' if the triplet can be inferred from the knowledge graph based on the reasoning paths and rules provided, otherwise return 'N'. Do not say anything else except your determination.
"""
