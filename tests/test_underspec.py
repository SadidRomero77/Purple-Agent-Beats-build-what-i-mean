"""Tests for underspec detection."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from purple_v2.underspec_detector import (
    detect_underspec_heuristic,
    patch_instruction_with_color,
    patch_instruction_with_count,
)


def test_fully_specified():
    result = detect_underspec_heuristic("Stack three red blocks on the green block.")
    assert not result.has_missing_color


def test_missing_color_no_colors():
    result = detect_underspec_heuristic("Stack blocks on the grid.")
    assert result.has_missing_color


def test_missing_color_multiple_colors():
    result = detect_underspec_heuristic(
        "Place a red block, then stack blocks behind the blue one."
    )
    assert result.has_missing_color


def test_missing_count():
    result = detect_underspec_heuristic("Stack red blocks on the grid.")
    assert result.has_missing_number


def test_patch_color():
    patched = patch_instruction_with_color("Stack blocks on the grid.", "Blue")
    assert "blue" in patched.lower()


def test_patch_count():
    patched = patch_instruction_with_count("Stack blocks on the grid.", 4)
    assert "4" in patched


def test_no_missing_with_explicit_count():
    result = detect_underspec_heuristic("Stack three red blocks on the grid.")
    assert not result.has_missing_number
