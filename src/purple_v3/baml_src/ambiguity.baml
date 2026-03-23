function DetectAmbiguity(instruction: string, start_structure: string, context_colors: string) -> AmbiguityResult {
  client GPT4oMini

  prompt #"
    You are an ambiguity detector for a block-building game on a 9x9 grid.

    Your job: determine if the instruction is MISSING a color or a count for any block placement.

    RULES FOR DETECTING MISSING COLOR:
    - If a phrase says to place/stack/build/add blocks WITHOUT specifying what color → missing_color = true
    - Examples of missing color:
      - "stack four blocks in front of them" (no color for those 4 blocks)
      - "place a block on top of the middle block" (no color for that block)
      - "Add a block to the shorter side" (no color specified)
      - "stack two blocks on top of the leftmost block" (no color for those blocks)
    - NOT missing color:
      - "stack five purple blocks" (purple is specified)
      - "place a red block" (red is specified)
      - "put a green block on top" (green is specified)

    RULES FOR DETECTING MISSING COUNT:
    - If a phrase says to place/stack/build blocks WITHOUT specifying how many → missing_count = true
    - Examples of missing count:
      - "stack red blocks directly in front" (how many red blocks?)
      - "Build a yellow stack" (how many in the stack?)
      - "Build a blue stack in front of the yellow one" (how many blue?)
    - NOT missing count:
      - "stack three red blocks" (three is specified)
      - "place a red block" (one is implied by "a")
      - "stack 4 blocks" (4 is specified)
      - "place a block on top" (one is implied by "a")

    CONTEXT:
    - Existing colors on the grid: {{ context_colors }}
    - If only one color exists and the instruction seems to continue the same pattern, inferred_color should be that color
    - For inferred_count, use the height of nearby stacks if visible, otherwise default to 3

    START_STRUCTURE: {{ start_structure }}
    INSTRUCTION: {{ instruction }}

    {{ ctx.output_format }}
  "#
}
