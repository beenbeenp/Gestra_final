# NEXT TASK

Task:
Live webcam end-to-end test and tuning.

Scope:
- Launch the game with `GESTRA_WEBCAM=1`:
  ```
  cd game_base/Street-Pyter
  GESTRA_WEBCAM=1 ../../.venv/bin/python main.py
  ```
- Stand full body in frame (~2.5m from camera)
- Test each action: idle, punch, kick, step forward, step backward
- Measure perceived latency (target <500ms)
- Check for idle jitter (false punch/kick triggers)

Tuning if needed:
- Adjust `HIP_VX_THRESHOLD` in `motion/pose_predictor.py` for forward/backward sensitivity
- Adjust `SMOOTHING_WINDOW` (currently 5) if actions feel sluggish or jittery
- If kick recall is too low in practice, consider retraining with more epochs or augmented data

Done when:
- 8/10 intended punches trigger in-game punch
- 7/10 intended kicks trigger in-game kick
- Idle stance produces no spurious attacks for 10+ seconds
- Forward/backward steps are visibly distinguishable
