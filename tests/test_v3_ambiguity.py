"""Tests for v3 ambiguity detection (heuristic fallback - no LLM needed)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v3.ambiguity_detector import (
    _detect_ambiguity_heuristic,
    patch_instruction_with_color,
    patch_instruction_with_count,
    _infer_color_from_context,
)
from purple_v3.grid import Grid


def test_color_ambiguity_detected():
    """Trial 1a: 'stack four blocks in front of them' — no color for second set."""
    instruction = "Stack five purple blocks in the middle of the grid, then stack four blocks in front of them."
    grid = Grid()
    result = _detect_ambiguity_heuristic(instruction, grid)
    assert result.has_missing_color is True


def test_no_ambiguity_fully_specified():
    """Fully specified instruction — no ambiguity."""
    instruction = "Stack three red blocks in the bottom right corner."
    grid = Grid()
    result = _detect_ambiguity_heuristic(instruction, grid)
    assert result.has_missing_color is False


def test_count_ambiguity_detected():
    """Trial 5: 'Then stack red blocks directly in front' — no count."""
    instruction = "Stack two yellow blocks on the middle square. Then stack red blocks directly in front of the green ones."
    grid = Grid()
    result = _detect_ambiguity_heuristic(instruction, grid)
    # Should detect missing count for "stack red blocks"
    assert result.has_missing_count is True


def test_patch_instruction_with_color():
    instruction = "stack four blocks in front of them"
    patched = patch_instruction_with_color(instruction, "Yellow")
    assert "yellow" in patched.lower()


def test_patch_instruction_with_count():
    instruction = "Build a yellow stack in front"
    patched = patch_instruction_with_count(instruction, 3)
    assert "3" in patched


def test_infer_color_single_color():
    """When only one color in instruction, infer that color."""
    grid = Grid()
    color = _infer_color_from_context("Stack five purple blocks, then stack blocks in front", grid)
    assert color == "Purple"


def test_infer_color_from_grid():
    """When no color in instruction, infer from grid."""
    grid = Grid.from_str("Red,0,50,0;Red,0,150,0;Blue,100,50,0")
    color = _infer_color_from_context("stack blocks on top", grid)
    assert color == "Red"  # most common


def test_trial_3a_color_under():
    """Trial 3a: 'place a block on top of the middle block' — no color."""
    instruction = "Place a blue block on the highlighted square. Place a blue block both to the left and to the right of that block. Then place a block on top of the middle block."
    grid = Grid()
    result = _detect_ambiguity_heuristic(instruction, grid)
    assert result.has_missing_color is True


def test_trial_17a_color_under():
    """Trial 17a: 'Stack two blocks on top of the leftmost block' — no color."""
    instruction = "Starting from the highlighted square, place a horizontal row of three yellow blocks going towards the left side of the grid. Stack two blocks on top of the leftmost block."
    grid = Grid()
    result = _detect_ambiguity_heuristic(instruction, grid)
    assert result.has_missing_color is True
