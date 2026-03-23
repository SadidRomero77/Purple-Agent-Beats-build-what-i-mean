"""Purple Agent v2 — Build What I Mean competition agent.

Skills-based pipeline: parse → detect underspec → analyze structure → plan (LLM)
→ deterministic fixes → execute → format → validate
"""
# Only import modules that don't require heavy external dependencies.
# Heavy imports (openai, a2a) are done lazily in server.py and build_planner.py.
from .grid import Grid, GridConfig, Block
from .instruction_parser import parse_green_message, ParsedInstruction
from .underspec_detector import detect_underspec_heuristic
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
