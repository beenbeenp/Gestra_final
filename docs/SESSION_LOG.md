# SESSION LOG

## 2026-04-22

### Base Game Setup
- Tested Street-Pyter as the base fighting game.
- Set up the Python 3.12 environment because pygame did not build correctly under Python 3.13.
- Confirmed the game could launch and both players could be controlled.
- Chose Street-Pyter because it was closer to the final goal than the earlier simple hand-gesture prototype.

### Input Hook Work
- Inspected the Player 1 input path in `main.py`, `lib.py`, and the fighter movement code.
- Added a small external-action path for Player 1 so webcam actions could be mapped into normal game inputs.
- Kept keyboard fallback available when webcam control is not active.
- Verified the edited game files compiled.

## 2026-04-26

### Webcam Control Prototype
- Added webcam control paths behind environment flags.
- Added an upper-body detector using MediaPipe pose landmarks.
- Mapped simple actions like punch, movement, idle, and block into Street-Pyter inputs.
- Tested the rule-based detector first so the game loop could be debugged before relying on a trained model.

## 2026-04-30

### Data and Model Pipeline
- Added the personal data recording flow for the five final classes:
  - `idle`
  - `lpunch`
  - `rpunch`
  - `forward`
  - `backward`
- Built the feature pipeline with 9 upper-body joints, shoulder-center normalization, shoulder-width scaling, velocity, and acceleration.
- Trained the TCN model using 20-frame windows and 81 features per frame.
- Kept the rule-based detector as the baseline for comparison.

## 2026-05-03

### Experiments
- Ran the main TCN training runs and saved results to `results/experiment_log.csv`.
- Logged the learning-rate, optimizer, batch-size, and window-length sweeps.
- Compared the rule-based detector against the TCN on the validation split.
- Recorded the final report values:
  - rule-based validation accuracy: 36.73%
  - TCN validation accuracy: 100.00%
  - sliding windows: 494

## 2026-05-05

### Figures and Report Assets
- Created `notebooks/reproduce_figures.ipynb` to regenerate the report figures.
- Saved generated figures to both `results/figures/` and `final_site/assets/`.
- Added the system pipeline, dataset summary, training curve, confusion matrix, rule-based vs TCN comparison, hyperparameter sweep, and resource summary figures.
- Added local screenshots for gameplay, pose overlay, quick-record, calibration, and the earlier Gesture-Game-main prototype.

## 2026-05-06

### Final Report Assembly
- Built the final report page in `final_site/index.html`.
- Organized the report around motivation, background, system overview, data, features, models, results, measured vs not measured, design evolution, ethics, future work, and references.
- Added the Gesture-Game-main origin story to explain how the project started before moving into Street-Pyter.
- Added the measured-vs-not-measured table to keep the claims clear.
- Added the ethics section about webcam data, local processing, consent, and accessibility limits.

## 2026-05-07

### Final Submission Check
- Checked that report figures and screenshots load from `final_site/assets/`.
- Confirmed the report keeps the model results scoped to a single-user controlled split.
- Confirmed the report does not claim multi-person generalization.
- Built `gestra_submission.zip` with `index.html`, `styles.css`, and `assets/` at the top level.

## 2026-05-08

### Final Cleanup
- Removed old fallback text from the report now that the screenshots are included.
- Kept all experiment values, figures, references, ethics, failure modes, and the Gesture-Game-main origin story unchanged.
- Rebuilt `gestra_submission.zip` from inside `final_site/` so `index.html` is at the zip top level.
