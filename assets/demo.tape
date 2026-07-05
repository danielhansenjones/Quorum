# Regenerate with: uv run vhs assets/demo.tape
# Renders the committed zero-key replay fixture (eval/fixtures/demo_replay.jsonl),
# so the recording is reproducible without a server, key, or Docker.
Output assets/demo.gif

Set FontSize 14
Set Width 1400
Set Height 820
Set Padding 12
Set PlaybackSpeed 1.0

Type "uv run python scripts/demo.py --replay --step 0.45"
Enter
Sleep 16s
