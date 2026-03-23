"""A2A server for Purple Agent v2 — Build What I Mean competition.

Pipeline: parse → detect underspec → analyze structure → plan (LLM)
        → deterministic fixes → execute → format → validate
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

from purple_v2.instruction_parser import parse_green_message, ParsedInstruction
from purple_v2.build_planner import BuildPlanner
from purple_v2.spatial_executor import SpatialExecutor, ExecutionError
from purple_v2.underspec_detector import (
    detect_underspec_heuristic,
    patch_instruction_with_color,
    patch_instruction_with_count,
)
from purple_v2.response_formatter import format_build_response, validate_build_response
from purple_v2.grid import Grid, GridConfig
from purple_v2.structure_analyzer import analyze_structure
from purple_v2.plan_verifier import (
    verify_plan,
    auto_fix_direction,
    auto_fix_each_end_caps,
    auto_fix_t_shape_extend,
)
from purple_v2.plan_patcher import patch_chain_references

logger = logging.getLogger(__name__)

# ── Fallback system prompt for direct LLM calls ──
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
    "- 'highlighted/middle square' = origin (0,0). Row from middle going right starts AT x=0.\n"
    "- 'each end' of row after extension → recalculate endpoints!\n"
    "- Chain refs: 'the green one' = most recently placed green, not original.\n"
    "- L/T shapes: extend horizontally, NEVER stack on top.\n"
)


def _make_agent_card(url: str) -> AgentCard:
    return AgentCard(
        name="PurpleAgentV2_BWIM",
        description="Spatial reasoning agent for Build What I Mean, powered by GPT-4o.",
        url=url,
        version="2.0.0",
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
    """Purple agent with skills-based build pipeline.

    Competitive advantages over AgentWhetters:
    1. GPT-4o (vs 4o-mini) for better spatial reasoning
    2. 18 adaptive prompt enrichment rules (vs 15)
    3. 12 worked examples in system prompt (vs 9)
    4. Self-verification pass before submitting
    5. Smarter underspec detection with cost-benefit analysis
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
        self._model = os.getenv("AGENT_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o"))
        # Strip provider prefix if present (e.g., "openai/gpt-4o" -> "gpt-4o")
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
        self._history: dict[str, list[dict]] = {}
        self._max_history = 5
        self._pending: dict[str, dict] = {}
        self._asked: set[str] = set()

    # ── Answer extraction ──

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
            logger.info("─── Input: %s", user_input[:300])

        if not self._api_key:
            await event_queue.enqueue_event(
                new_agent_text_message(
                    "[BUILD]", context_id=context.context_id
                )
            )
            return

        parsed = parse_green_message(user_input, self._config)

        # Handle feedback / transition messages
        if parsed.is_feedback:
            # On new task (new seed), clear ALL state for a fresh start
            if "new task is starting" in parsed.feedback_text.lower():
                self._history.pop(ctx_id, None)
                self._pending.pop(ctx_id, None)
                self._asked.discard(ctx_id)
            else:
                self._add_history(ctx_id, "feedback", parsed.feedback_text)
                self._pending.pop(ctx_id, None)
                self._asked.discard(ctx_id)
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

        # Hard guard: never [ASK] twice per round
        if response.startswith("[ASK]") and ctx_id in self._asked:
            logger.warning("HARD GUARD: suppressing repeated [ASK]")
            response = await self._direct_llm_call(user_input, ctx_id)
        elif response.startswith("[ASK]"):
            self._asked.add(ctx_id)

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
            color_str = answered_colors[0] if answered_colors else "Purple"

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

        if response is None:
            hint = f"\n\nThe answer is: {color_str}"
            if answered_count is not None:
                hint += f", {answered_count} blocks"
            hint += ". Use this info. Respond with [BUILD]."
            response = await self._direct_llm_call(orig_input + hint, ctx_id)

        return response

    # ── Skills pipeline ──

    async def _skills_pipeline(
        self,
        parsed: ParsedInstruction,
        ctx_id: str,
        original_input: str = "",
        override_count: int | None = None,
    ) -> str | None:
        """Full pipeline: underspec → analyze → plan → fix → execute → format."""
        try:
            # Step 1: Pre-LLM underspec detection
            heuristic = detect_underspec_heuristic(parsed.instruction_text)
            logger.info("Heuristic: %s", heuristic.details)
            inferred_count = override_count or heuristic.inferred_count or 3

            # Compound ask (both color and count missing)
            if (
                heuristic.has_missing_color
                and heuristic.has_missing_number
                and ctx_id not in self._asked
            ):
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "ask_type": "compound",
                    "uncounted_color": heuristic.uncounted_color,
                }
                q = heuristic.suggested_compound_question or (
                    "What color should the unspecified blocks be, "
                    "and how many blocks should be in that stack?"
                )
                return f"[ASK];{q}"

            # Color-only ask
            if heuristic.has_missing_color and ctx_id not in self._asked:
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "ask_type": "color",
                }
                q = heuristic.suggested_question or "What color should the unspecified blocks be?"
                return f"[ASK];{q}"

            # Count-only ask
            if (
                heuristic.has_missing_number
                and not heuristic.has_missing_color
                and ctx_id not in self._asked
                and override_count is None
            ):
                self._pending[ctx_id] = {
                    "parsed": parsed,
                    "original_input": original_input,
                    "ask_type": "count",
                    "uncounted_color": heuristic.uncounted_color,
                }
                q = heuristic.suggested_count_question or "How many blocks should be in the stack?"
                return f"[ASK];{q}"

            # Already asked — fill with inferred color
            if heuristic.has_missing_color:
                fill = heuristic.inferred_color or "Purple"
                parsed.instruction_text = patch_instruction_with_color(
                    parsed.instruction_text, fill
                )

            # Step 2: Analyze existing structure
            structure_info = analyze_structure(parsed.start_grid)
            logger.info("Structure: %s", structure_info.describe()[:200])

            # Step 3: Plan via LLM
            steps = await self._planner.decompose(
                parsed.instruction_text,
                parsed.start_grid,
                parsed.speaker,
                structure_hint=structure_info.describe(),
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
                    s.color = heuristic.inferred_color or "Purple"
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

            for entry in self._history.get(ctx_id, [])[-self._max_history * 3 :]:
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
                # Try to find [BUILD] somewhere in the response
                idx = content.find("[BUILD]")
                if idx >= 0:
                    content = content[idx:]
                else:
                    content = "[BUILD]"

            return content
        except Exception as exc:
            logger.warning("Fallback LLM failed: %s", exc)
            return "[BUILD]"

    # ── History ──

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
    parser = argparse.ArgumentParser(description="Purple Agent v2 — Build What I Mean")
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

    logger.info("Starting Purple Agent v2 on %s:%d", args.host, args.port)
    app = A2AStarletteApplication(agent_card=card, http_handler=handler)
    uvicorn.run(app.build(), host=args.host, port=args.port, timeout_keep_alive=300)


if __name__ == "__main__":
    main()
