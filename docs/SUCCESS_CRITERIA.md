# SUCCESS CRITERIA

## Technical success
- system runs for 1 minute without crashing
- average real-time FPS >= 20
- pose extraction is stable enough for full-body tracking

## Model success
- 5-class balanced accuracy >= 75%
- punch precision >= 80%
- kick precision >= 80%

## Interaction success
- 8/10 intended punches trigger correct punch action
- 7/10 intended kicks trigger correct kick action
- idle false triggers remain low
- forward and backward are visibly distinguishable in live play

## Course success
- can explain how data was collected
- can explain labeling rules
- can explain failure modes
- can explain why LSTM is used