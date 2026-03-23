// Ambiguity detection result
class AmbiguityResult {
  has_missing_color bool @description("True if any block placement lacks a specified color")
  has_missing_count bool @description("True if any block placement lacks a specified count/number")
  missing_color_phrase string @description("The phrase that is missing a color, or empty string")
  missing_count_phrase string @description("The phrase that is missing a count, or empty string")
  suggested_color_question string @description("Question to ask about the missing color, or empty string")
  suggested_count_question string @description("Question to ask about the missing count, or empty string")
  inferred_color string @description("Best guess color if we had to guess, based on context")
  inferred_count int @description("Best guess count if we had to guess, based on context (default 3)")
  reasoning string @description("Brief explanation of why ambiguity was or was not detected")
}

// Build step for the planner
enum BuildAction {
  Stack
  Place
  PlaceRelative
  ExtendRow
  PlaceAtCorners
}

class Position {
  x int @description("X coordinate. Valid: -400,-300,-200,-100,0,100,200,300,400")
  z int @description("Z coordinate. Valid: -400,-300,-200,-100,0,100,200,300,400")
}

enum Direction {
  Right
  Left
  Front
  Behind
}

class BuildStep {
  action BuildAction
  color string @description("Block color (capitalized). Use 'Uncolored' if not specified in instruction.")
  count int @description("Number of blocks. Use 0 if not specified in instruction (means 'Uncounted').")
  position Position @description("Where to place blocks (x,z). Y is auto-computed.")
  direction Direction? @description("Direction for extend_row or place_relative. Null if not applicable.")
  reference string? @description("Reference to another block for place_relative, e.g. 'Red_block_at_0_0'. Null if not applicable.")
}

class BuildPlan {
  steps BuildStep[] @description("Ordered list of atomic build steps")
}
