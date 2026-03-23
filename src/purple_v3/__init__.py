"""Purple Agent v3 — Build What I Mean competition agent.

BAML-powered pipeline: parse → LLM ambiguity detect → analyze structure → LLM plan (BAML)
→ deterministic fixes → execute → format → validate

Key improvements over v2:
1. LLM-based ambiguity detection via BAML (replaces broken regex heuristics)
2. Worked examples from actual stimulus data in planner prompt
3. History context wired into planner LLM calls
4. GPT-4o-mini for cost efficiency
5. Ask on ANY detected ambiguity
"""
from .grid import Grid, GridConfig, Block
from .instruction_parser import parse_green_message, ParsedInstruction
from .structure_analyzer import analyze_structure, StructureInfo
from .plan_verifier import (
    verify_plan,
    auto_fix_direction,
    auto_fix_each_end_caps,
    auto_fix_t_shape_extend,
)
from .plan_patcher import patch_chain_references
from .response_formatter import format_build_response, validate_build_response
from .prompt_enricher import enrich_prompt
