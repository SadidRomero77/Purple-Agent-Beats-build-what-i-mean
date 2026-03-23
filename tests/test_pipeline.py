"""Tests for the full pipeline components working together."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v2.grid import Grid
from purple_v2.instruction_parser import parse_green_message
from purple_v2.structure_analyzer import analyze_structure
from purple_v2.response_formatter import format_build_response, validate_build_response
from purple_v2.prompt_enricher import enrich_prompt
from purple_v2.plan_verifier import auto_fix_direction
from purple_v2.build_planner import BuildStep


def test_parse_instruction():
    msg = (
        "[TASK_DESCRIPTION] Grid: 9x9 cells.\n"
        "[SPEAKER] Anna\n"
        "[START_STRUCTURE] Red,0,50,0\n"
        "Stack three blue blocks on top of the red block."
    )
    parsed = parse_green_message(msg)
    assert parsed.speaker == "Anna"
    assert parsed.start_structure == "Red,0,50,0"
    assert "Stack three blue blocks" in parsed.instruction_text


def test_parse_feedback():
    parsed = parse_green_message("Feedback: Correct structure! +10 points.")
    assert parsed.is_feedback


def test_parse_answer():
    parsed = parse_green_message("Answer: Blue (-5 points for asking)")
    assert not parsed.is_feedback
    assert "Answer:" in parsed.instruction_text


def test_structure_analyzer_line():
    g = Grid.from_str("Red,-100,50,0;Red,0,50,0;Red,100,50,0")
    info = analyze_structure(g)
    assert len(info.lines) == 1
    assert info.lines[0]["count"] == 3


def test_structure_analyzer_stack():
    g = Grid.from_str("Blue,0,50,0;Blue,0,150,0;Blue,0,250,0")
    info = analyze_structure(g)
    assert len(info.stacks) >= 1


def test_structure_analyzer_empty():
    g = Grid()
    info = analyze_structure(g)
    assert info.describe() == "The grid is empty."


def test_format_build_response():
    g = Grid.from_str("Red,0,50,0;Blue,100,50,200")
    response = format_build_response(g)
    assert response.startswith("[BUILD]")
    assert "Red,0,50,0" in response
    assert "Blue,100,50,200" in response


def test_validate_good_response():
    response = "[BUILD];Red,0,50,0;Blue,100,150,200"
    valid, errors = validate_build_response(response)
    assert valid
    assert errors == []


def test_validate_bad_coordinates():
    response = "[BUILD];Red,0,50,0;Blue,150,50,200"  # 150 is invalid x
    valid, errors = validate_build_response(response)
    assert not valid


def test_enrichment_in_front():
    text = enrich_prompt("Place a block in front of the red one.")
    assert "+z" in text


def test_enrichment_corner():
    text = enrich_prompt("Place blocks in each corner.")
    assert "corner" in text.lower()


def test_enrichment_no_match():
    text = enrich_prompt("Hello world")
    assert text == ""


def test_auto_fix_direction():
    steps = [BuildStep(action="extend_row", color="Red", count=3,
                       position={"x": 0, "z": 0}, direction="right")]
    fixed = auto_fix_direction("extend the row going left", steps)
    assert fixed[0].direction == "left"


def test_auto_fix_direction_no_change():
    steps = [BuildStep(action="extend_row", color="Red", count=3,
                       position={"x": 0, "z": 0}, direction="right")]
    fixed = auto_fix_direction("extend the row going right", steps)
    assert fixed[0].direction == "right"
