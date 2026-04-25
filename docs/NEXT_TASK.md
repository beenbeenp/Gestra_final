# NEXT TASK

Task:
Run a one-person pilot pose-capture pass, not full dataset collection.

Scope:
- Use `motion/pose_capture.py`
- Stand full body in frame
- Save a few short landmark samples for:
  - idle
  - forward
  - backward
  - punch
  - kick

Constraints:
- do not connect pose output to the game yet
- do not train a model yet
- do not collect the full dataset yet
- keep the camera setup consistent during the pilot

Done when:
- each action has at least one short saved JSON sample
- saved samples contain nonzero MediaPipe pose landmark frames
- camera framing and lighting are judged good enough for a later dataset pass
