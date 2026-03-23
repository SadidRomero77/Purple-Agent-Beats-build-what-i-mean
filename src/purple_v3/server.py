"""A2A server for Purple Agent v3 — Build What I Mean competition.

BAML-powered pipeline:
  parse → LLM ambiguity detect → analyze structure → LLM plan (BAML)
  → deterministic fixes → execute → format → validate

Key improvements over v2:
1. LLM-based ambiguity detection (BAML) — always fires on real instructions
2. History wired into planner — LLM learns across rounds
3. GPT-4o-mini — cost-effective, good enough with deterministic pipeline
4. Ask on ANY ambiguity — maximize information
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message
from openai import AsyncOpenAI, AsyncAzureOpenAI
import uvicorn

from purple_v3.instruction_parser import parse_green_message, ParsedInstruction
from purple_v3.build_planner import BuildPlanner, BuildStep
from purple_v3.spatial_executor import SpatialExecutor, ExecutionError
from purple_v3.ambiguity_detector import (
    detect_ambiguity_llm,
    _infer_color_from_context,
    patch_instruction_with_color,
    patch_instruction_with_count,
)
from purple_v3.response_formatter import format_build_response, validate_build_response
from purple_v3.grid import Grid, GridConfig
from purple_v3.structure_analyzer import analyze_structure
from purple_v3.plan_verifier import (
    verify_plan,
    auto_fix_direction,
    auto_fix_each_end_caps,
    auto_fix_t_shape_extend,
)
from purple_v3.plan_patcher import patch_chain_references

logger = logging.getLogger(__name__)

_FALLBACK_SYSTEM_PROMPT = (
    "You are a block-building agent on a 9x9 grid.\n\n"
    "GRID COORDINATES:\n"
    "- The grid is the x-z plane. Origin (0,0) is the center.\n"
    "- Valid x,z: [-400,-300,-200,-100,0,100,200,300,400]\n"
    "- Y is vertical. Ground=50. Each block +100. Valid y: [50,150,250,350,450]\n"
    "- Format: Color,x,y,z (e.g., Red,0,50,0)\n\n"
    "DIRECTIONS:\n"
    "- 'in front of' = +z | 'behind' = -z\n"
    "- 'to the right' = +x | 'to the left' = -x\n"
    "- 'on top of' = +y (same x,z)\n\n"
    "CORNERS (y=50):\n"
    "- bottom-left=(-400,50,400), bottom-right=(400,50,400)\n"
    "- top-left=(-400,50,-400), top-right=(400,50,-400)\n\n"
    "RESPONSE: [BUILD];Color,x,y,z;Color,x,y,z;...\n"
    "Include ALL blocks (existing + new). No spaces. Colors capitalized.\n"
    "NEVER respond with [ASK] — always BUILD your best guess.\n\n"
    "STRATEGY:\n"
    "- Unspecified color → reuse context or match adjacent blocks.\n"
    "- Unspecified count → match adjacent stack height or use 3.\n"
    "- Include START_STRUCTURE blocks plus new blocks.\n"
    "- 'highlighted/middle square' = origin (0,0).\n"
    "- 'each end' of row after extension → recalculate endpoints!\n"
    "- Chain refs: 'the green one' = most recently placed green.\n"
    "- L/T shapes: extend horizontally, NEVER stack on top.\n"
)


def _make_agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="PurpleAgentV3_BWIM",
        description="BAML-powered spatial reasoning agent for Build What I Mean.",
        url=url,
        version="3.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[
            AgentSkill(
                id="block_building",
                name="Block building",
                description="Build block structures on a 3D grid following natural language instructions",
                tags=["blocks", "building", "spatial"],
                examples=[],
            )
        ],
    )


def _make_openai_client(api_key: str, base_url: str | None = None) -> AsyncOpenAI:
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
    if azure_endpoint:
        api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
        return AsyncAzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


class PurpleAgent(AgentExecutor):
    """Purple agent v3 with BAML-powered pipeline.

    Competitive advantages:
    1. LLM-based ambiguity detection (BAML) — never misses underspec
    2. GPT-4o-mini with deterministic execution pipeline
    3. History wired into planner for cross-round learning
    4. 18 adaptive prompt enrichment rules + 9 worked examples
    5. Self-verification pass with auto-corrections
    6. Ask on ANY ambiguity detected
    """

    _ANSWER_RE = re.compile(
        r"^Answer:\s*(.+?)(?:\s*\(.*points.*\))?$", re.IGNORECASE
    )
    _COLOR_NAMES = {
        "red", "blue", "green", "yellow", "purple", "orange",
        "white", "black", "brown", "pink", "grey", "gray", "cyan",
    }
    _WORD_TO_INT = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }

    def __init__(self, debug: bool = False):
        self._debug = debug
        self._model = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
        if "/" in self._model:
            self._model = self._model.split("/", 1)[1]
        self._api_key = os.getenv("OPENAI_API_KEY", "").strip()
        self._base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        self._client = _make_openai_client(self._api_key, self._base_url)
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()
        if azure_deployment:
            self._model = azure_deployment
        self._config = GridConfig()
        self._planner = BuildPlanner(self._client, self._model, self._config)
        # Per-context state
        self._history: dict[str, list[dict]] = {}
        self._max_history = 8  # Balanced: enough context, no contamination
        self._pending: dict[str, dict] = {}
        self._asked: set[str] = set()
        # ASK rate tracking: hard cap at 20%
        self._round_count: dict[str, int] = {}
        self._ask_count: dict[str, int] = {}
        # Cross-round feedback cache: maps instruction pattern → correction
        self._feedback_cache: dict[str, str] = {}

    # ── Answer extraction ──

    @classmethod
    def _count_instruction_colors(cls, text: str) -> int:
        """Count unique colors mentioned in instruction text."""
        lower = text.lower()
        return sum(1 for c in cls._COLOR_NAMES if re.search(r'\b' + c + r'\b', lower))

    @classmethod
    def _extract_answer_colors(cls, text: str) -> list[str]:
        m = cls._ANSWER_RE.match(text.strip())
        if not m:
            return []
        body = m.group(1).strip().rstrip(".,!").lower()
        colors: list[str] = []
        for match in re.finditer(
            r"\b(" + "|".join(cls._COLOR_NAMES) + r")\b", body
        ):
            c = match.group(1).capitalize()
            if c not in colors:
                colors.append(c)
        return colors

    @classmethod
    def _extract_answer_count(cls, text: str) -> int | None:
        m = cls._ANSWER_RE.match(text.strip())
        if not m:
            return None
        body = m.group(1).strip().rstrip(".,!").lower()
        digit_m = re.search(r"\b(\d+)\b", body)
        if digit_m:
            return int(digit_m.group(1))
        for word, val in cls._WORD_TO_INT.items():
            if re.search(r"\b" + word + r"\b", body):
                return val
        return None

    # ── Main execute ──

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        ctx_id = context.context_id or "default"

        if self._debug:
            logger.info("─── Input [%s]: %s", ctx_id, user_input[:300])

        if not self._api_key:
            await event_queue.enqueue_event(
                new_agent_text_message("[BUILD]", context_id=context.context_id)
            )
            return

        parsed = parse_green_message(user_input, self._config)

        # Handle feedback / transition messages
        if parsed.is_feedback:
            # Cache negative feedback for cross-round learning
            feedback_lower = parsed.feedback_text.lower()
            if "incorrect" in feedback_lower or "-10" in feedback_lower:
                # Get last instruction for this context and cache the failure
                last_instructions = [
                    e for e in self._history.get(ctx_id, [])
                    if e["type"] == "instruction"
                ]
                if last_instructions:
                    key = self._instruction_key(last_instructions[-1]["content"])
                    self._feedback_cache[key] = (
                        f"PREVIOUS ATTEMPT FAILED on similar instruction. "
                        f"Feedback was: {parsed.feedback_text[:200]}. "
                        f"Try a different approach."
                    )
            # Add round separator to prevent history contamination
            self._add_history(ctx_id, "feedback", parsed.feedback_text)
            self._add_history(ctx_id, "separator", "--- NEW ROUND ---")
            self._pending.pop(ctx_id, None)
            self._asked.discard(ctx_id)
            # Track round count for ASK cap
            self._round_count[ctx_id] = self._round_count.get(ctx_id, 0) + 1
            await event_queue.enqueue_event(
                new_agent_text_message("[BUILD]", context_id=context.context_id)
            )
            return

        # ── Check if this is an answer to our question ──
        pending = self._pending.pop(ctx_id, None)
        ask_type = pending.get("ask_type", "color") if pending else "color"
        answered_colors = self._extract_answer_colors(parsed.instruction_text)
        answered_count = self._extract_answer_count(parsed.instruction_text)

        if pending and (answered_colors or answered_count is not None):
            self._asked.add(ctx_id)
            response = await self._handle_answer(
                pending, ask_type, answered_colors, answered_count, ctx_id, user_input
            )
        else:
            response = await self._skills_pipeline(parsed, ctx_id, user_input)
            if response is None:
                logger.info("Skills pipeline failed → fallback LLM")
                response = await self._direct_llm_call(user_input, ctx_id)

        # Hard guard: never [ASK] twice per context
        if response.startswith("[ASK]") and ctx_id in self._asked:
            logger.warning("HARD GUARD: suppressing repeated [ASK]")
            response = await self._direct_llm_call(user_input, ctx_id)
        elif response.startswith("[ASK]"):
            self._asked.add(ctx_id)
            self._ask_count[ctx_id] = self._ask_count.get(ctx_id, 0) + 1

        # Track round (non-feedback, non-answer messages)
        if not parsed.is_feedback and not (pending and (answered_colors or answered_count is not None)):
            self._round_count[ctx_id] = self._round_count.get(ctx_id, 0) + 1

        self._add_history(ctx_id, "instruction", parsed.instruction_text)
        self._add_history(ctx_id, "response", response)

        await event_queue.enqueue_event(
            new_agent_text_message(response, context_id=context.context_id)
        )

    # ── Answer handling ──

    async def _handle_answer(
        self,
        pending: dict,
        ask_type: str,
        answered_colors: list[str],
        answered_count: int | None,
        ctx_id: str,
        original_input: str,
    ) -> str:
        orig_parsed = pending["parsed"]
        orig_input = pending.get("original_input", original_input)

        # Disambiguate colors
        instruction_colors = {
            c for c in self._COLOR_NAMES if c in orig_parsed.instruction_text.lower()
        }
        if len(answered_colors) > 1:
            new_colors = [c for c in answered_colors if c.lower() not in instruction_colors]
            color_str = new_colors[0] if new_colors else answered_colors[-1]
        else:
            color_str = answered_colors[0] if answered_colors else "Red"

        if ask_type == "compound":
            patched = patch_instruction_with_color(orig_parsed.instruction_text, color_str)
            if answered_count is not None:
                patched = patch_instruction_with_count(patched, answered_count)
                orig_parsed._answered_count = answered_count
            orig_parsed.instruction_text = patched
            response = await self._skills_pipeline(
                orig_parsed, ctx_id, orig_input, override_count=answered_count
            )
        elif ask_type == "count" and answered_count is not None:
            patched = patch_instruction_with_count(
                orig_parsed.instruction_text,
                answered_count,
                target_color=pending.get("uncounted_color", ""),
            )
            orig_parsed.instruction_text = patched
            orig_parsed._answered_count = answered_count
            response = await self._skills_pipeline(
                orig_parsed, ctx_id, orig_input, override_count=answered_count
            )
        else:
            patched = patch_instruction_with_color(orig_parsed.instruction_text, color_str)
            orig_parsed.instruction_text = patched
            response = await self._skills_pipeline(orig_parsed, ctx_id, orig_input)

        # Re-verification guard: if pipeline produced a response, check
        # that block count is reasonable vs start structure
        if response and response.startswith("[BUILD]"):
            start_count = len(orig_parsed.start_grid.blocks)
            response_blocks = response.count(";")
            # If response has way too many or too few blocks vs expected,
            # fall back to direct LLM with the answer hint
            if response_blocks > 0 and (response_blocks > start_count + 15 or response_blocks < start_count):
                logger.warning(
                    "Re-verification: response has %d blocks vs %d start — suspicious, using fallback",
                    response_blocks, start_count
                )
                response = None

        if response is None:
            hint = f"\n\nThe answer is: {color_str}"
            if answered_count is not None:
                hint += f", {answered_count} blocks"
            hint += ". Use this info. Respond with [BUILD]."
            response = await self._direct_llm_call(orig_input + hint, ctx_id)

        return response

    # ── Skills pipeline (with BAML + history) ──

    async def _skills_pipeline(
        self,
        parsed: ParsedInstruction,
        ctx_id: str,
        original_input: str = "",
        override_count: int | None = None,
    ) -> str | None:
        """Full pipeline: LLM ambiguity → analyze → BAML plan → fix → execute → format."""
        try:
            # Step 1: LLM-based ambiguity detection (BAML)
            ambiguity = await detect_ambiguity_llm(
                parsed.instruction_text,
                parsed.start_grid,
            )
            logger.info("Ambiguity: color=%s count=%s reason=%s",
                         ambiguity.has_missing_color, ambiguity.has_missing_count,
                         ambiguity.reasoning[:100])

            inferred_count = override_count or ambiguity.inferred_count or 3

            # ── ASK strategy (EV-based) ──
            # NEVER ask about count: -5 penalty rarely justified, infer instead.
            # Only ask about color when there's genuine ambiguity:
            #   - Instruction has 0 colors AND grid has 2+ colors (can't guess)
            #   - Instruction has 2+ colors AND colorless phrase (unclear which)
            # Single-color instructions: just reuse that color (don't ask).

            # ── Hard ASK cap: never exceed 20% ask rate ──
            rounds = self._round_count.get(ctx_id, 0)
            asks = self._ask_count.get(ctx_id, 0)
            ask_rate_exceeded = rounds >= 5 and asks / max(rounds, 1) > 0.20

            should_ask_color = False
            if ambiguity.has_missing_color and ctx_id not in self._asked and not ask_rate_exceeded:
                instruction_colors = self._count_instruction_colors(parsed.instruction_text)
                grid_colors = len(parsed.start_grid.unique_colors()) if parsed.start_grid.blocks else 0

                if instruction_colors == 0 and grid_colors >= 2:
                    # No color clue at all, grid has multiple — must ask
                    should_ask_color = True
                elif instruction_colors >= 2:
                    # Multiple colors mentioned, unclear which applies — ask
                    should_ask_color = True
                # Single color in instruction or single grid color → infer, don't ask

            if should_ask_color:
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "ask_type": "color",
                }
                return f"[ASK];{ambiguity.suggested_color_question}"

            # Fill missing color with inferred value (never ask)
            if ambiguity.has_missing_color:
                fill = ambiguity.inferred_color or _infer_color_from_context(
                    parsed.instruction_text, parsed.start_grid
                )
                parsed.instruction_text = patch_instruction_with_color(
                    parsed.instruction_text, fill
                )

            # Fill missing count with inferred value (never ask about count)

            # Step 2: Analyze existing structure
            structure_info = analyze_structure(parsed.start_grid)
            logger.info("Structure: %s", structure_info.describe()[:200])

            # Step 3: Plan via BAML LLM (with history!)
            history = self._history.get(ctx_id, [])

            # Check feedback cache for similar past failures
            ikey = self._instruction_key(parsed.instruction_text)
            cached_correction = self._feedback_cache.get(ikey, "")

            steps = await self._planner.decompose(
                parsed.instruction_text,
                parsed.start_grid,
                parsed.speaker,
                structure_hint=structure_info.describe(),
                correction_hint=cached_correction,
                history=history,
            )
            if not steps:
                return None

            logger.info("Planner: %d steps", len(steps))

            # Step 3b: Chain reference patching
            steps = patch_chain_references(steps, parsed.start_grid)

            # Step 3c: Auto-fixes (deterministic)
            steps = auto_fix_direction(parsed.instruction_text, steps)
            steps = auto_fix_each_end_caps(parsed.instruction_text, steps, parsed.start_grid)
            steps = auto_fix_t_shape_extend(parsed.instruction_text, steps, parsed.start_grid)

            # Step 3d: Verify plan
            verification = verify_plan(
                parsed.instruction_text, steps, len(parsed.start_grid.blocks)
            )
            if verification.has_critical:
                logger.info("Re-planning due to: %s", verification.correction_prompt()[:200])
                steps = await self._planner.decompose(
                    parsed.instruction_text,
                    parsed.start_grid,
                    parsed.speaker,
                    structure_hint=structure_info.describe(),
                    correction_hint=verification.correction_prompt(),
                    history=history,
                )
                if not steps:
                    return None
                steps = patch_chain_references(steps, parsed.start_grid)
                steps = auto_fix_direction(parsed.instruction_text, steps)
                steps = auto_fix_each_end_caps(parsed.instruction_text, steps, parsed.start_grid)
                steps = auto_fix_t_shape_extend(parsed.instruction_text, steps, parsed.start_grid)

            # Step 4: Resolve uncolored/uncounted
            _UNCOLORED = {"uncolored", "unknown", "unspecified", "?"}
            for s in steps:
                if s.color.lower() in _UNCOLORED:
                    s.color = ambiguity.inferred_color or "Purple"
                if isinstance(s.count, str) and s.count.lower() in (
                    "uncounted", "unknown", "unspecified", "?"
                ):
                    s.count = inferred_count

            # Step 5: Execute deterministically
            exec_grid = Grid.from_str(parsed.start_grid.to_str(), config=self._config)
            executor = SpatialExecutor(exec_grid)
            executor.execute_plan(steps)

            # Step 6: Format response
            response = format_build_response(exec_grid)

            # Step 7: Validate
            is_valid, errors = validate_build_response(response, self._config)
            if not is_valid:
                logger.warning("Validation failed: %s", errors)
                return None

            logger.info("Pipeline OK: %d blocks", len(exec_grid.blocks))
            return response

        except ExecutionError as exc:
            logger.warning("Execution error: %s", exc)
            return None
        except Exception as exc:
            logger.warning("Pipeline error: %s", exc)
            return None

    # ── Fallback LLM ──

    async def _direct_llm_call(self, user_input: str, ctx_id: str) -> str:
        try:
            messages: list[dict] = [{"role": "system", "content": _FALLBACK_SYSTEM_PROMPT}]

            # Include history in fallback too!
            for entry in self._history.get(ctx_id, [])[-self._max_history * 3:]:
                if entry["type"] == "instruction":
                    messages.append({"role": "user", "content": entry["content"]})
                elif entry["type"] == "response":
                    messages.append({"role": "assistant", "content": entry["content"]})
                elif entry["type"] == "feedback":
                    messages.append({"role": "user", "content": entry["content"]})

            messages.append({"role": "user", "content": user_input})

            completion = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.15,
                max_tokens=2048,
            )
            content = (completion.choices[0].message.content or "").strip()

            if content.startswith("[ASK]"):
                logger.warning("Fallback LLM tried [ASK] → empty [BUILD]")
                content = "[BUILD]"
            elif not content.startswith("[BUILD]"):
                idx = content.find("[BUILD]")
                if idx >= 0:
                    content = content[idx:]
                else:
                    content = "[BUILD]"

            return content
        except Exception as exc:
            logger.warning("Fallback LLM failed: %s", exc)
            return "[BUILD]"

    # ── History & Learning ──

    @staticmethod
    def _instruction_key(instruction: str) -> str:
        """Create a normalized key for instruction pattern matching.

        Strips numbers and specific colors to match structurally similar
        instructions across different rounds/lists.
        """
        lower = instruction.lower().strip()
        # Remove numbers
        key = re.sub(r'\b\d+\b', 'N', lower)
        # Remove specific color names
        for c in ("red", "blue", "green", "yellow", "purple", "orange",
                   "white", "black", "brown", "pink", "grey", "gray", "cyan"):
            key = re.sub(r'\b' + c + r'\b', 'COLOR', key)
        # Collapse whitespace
        key = re.sub(r'\s+', ' ', key).strip()
        return key

    def _add_history(self, ctx_id: str, entry_type: str, content: str) -> None:
        if ctx_id not in self._history:
            self._history[ctx_id] = []
        self._history[ctx_id].append({"type": entry_type, "content": content})
        max_entries = self._max_history * 3
        if len(self._history[ctx_id]) > max_entries:
            self._history[ctx_id] = self._history[ctx_id][-max_entries:]

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Purple Agent v3 — Build What I Mean (BAML)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9018)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--card-url", default="")
    args = parser.parse_args()

    debug = args.debug or os.getenv("AGENT_DEBUG", "").lower() in ("1", "true", "yes")
    logging.basicConfig(level=logging.INFO if debug else logging.WARNING)

    card_url = args.card_url
    if not card_url:
        host = "127.0.0.1" if args.host == "0.0.0.0" else args.host
        card_url = f"http://{host}:{args.port}"

    card = _make_agent_card(card_url)
    handler = DefaultRequestHandler(
        agent_executor=PurpleAgent(debug=debug),
        task_store=InMemoryTaskStore(),
    )

    logger.info("Starting Purple Agent v3 on %s:%d", args.host, args.port)
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
