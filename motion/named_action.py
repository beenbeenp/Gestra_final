"""Map a small action vocabulary to the nested input list that Street-Pyter expects.

Street-Pyter's `Character.get_input()` returns a list of two lists:
    [[up, down, left, right], [lpunch, mpunch, hpunch, lkick, mkick, hkick]]

We collapse motion-control to five named actions and map each onto that shape.
The motion side speaks strings; the game side never touches strings.

The choice of which kick / punch tier to fire is fixed for v1:
    "punch" -> medium punch
    "kick"  -> medium kick
"""

ACTION_NAMES = ("idle", "lpunch", "rpunch", "forward", "backward")


def _empty_input():
    return [
        [False, False, False, False],
        [False, False, False, False, False, False],
    ]


def named_to_input(name, *, flip=False):
    inp = _empty_input()
    if name is None or name == "idle":
        return inp

    if name == "lpunch":
        inp[1][0] = True
        return inp

    if name == "rpunch":
        inp[1][1] = True
        return inp

    if name == "punch":
        inp[1][1] = True
        return inp

    if name == "forward":
        if flip:
            inp[0][2] = True
        else:
            inp[0][3] = True
        return inp

    if name == "backward":
        if flip:
            inp[0][3] = True
        else:
            inp[0][2] = True
        return inp

    raise ValueError(f"unknown action name: {name!r}")
