"""Win probability for rally-point sports (badminton, volleyball) with a Markov chain.

Idea: if side A wins each rally with probability p (independently), the chance of winning a game or set
from any score can be worked out exactly, because the only thing that matters is the current score.
That is a Markov chain: the next state depends only on the present state.

    P(a, b) = p * P(a + 1, b) + (1 - p) * P(a, b + 1)

with the end states fixed: a reaches the target with a 2-point lead (win, P = 1) or b does (loss, P = 0).
Level at target-1 each ('deuce') has a closed form: p^2 / (p^2 + (1-p)^2).
Badminton games stop at 30 (29-all: next point wins). Volleyball sets have no cap.
"""
from functools import lru_cache

import numpy as np

RULES = {
    "badminton": {"target": 21, "cap": 30, "deciding_target": 21, "best_of": 3},
    "volleyball": {"target": 25, "cap": None, "deciding_target": 15, "best_of": 5},
}


@lru_cache(maxsize=200_000)
def game(a, b, p, target, cap=None):
    """Chance that side A wins the game/set from score a:b (A wins each rally with chance p)."""
    p = min(max(p, 1e-6), 1 - 1e-6)
    if cap is not None:
        if a >= cap or (a >= target and a - b >= 2):
            return 1.0
        if b >= cap or (b >= target and b - a >= 2):
            return 0.0
        if a == b == cap - 1:
            return p
    else:
        if a >= target and a - b >= 2:
            return 1.0
        if b >= target and b - a >= 2:
            return 0.0
    if a >= target - 1 and b >= target - 1 and cap is None:
        q = 1 - p
        deuce = p * p / (p * p + q * q)
        if a == b:
            return deuce
        return p + q * deuce if a > b else p * deuce
    return p * game(a + 1, b, p, target, cap) + (1 - p) * game(a, b + 1, p, target, cap)


def match(sets_a, sets_b, p, sport, a=0, b=0):
    """Chance that A wins the match, from sets_a:sets_b in games and a:b in the current game."""
    r = RULES[sport]
    need = r["best_of"] // 2 + 1
    p = float(round(p, 4))                 # rounding keeps the cache small and changes nothing visible

    @lru_cache(maxsize=None)
    def from_sets(sa, sb):
        if sa >= need:
            return 1.0
        if sb >= need:
            return 0.0
        deciding = sa == sb == need - 1
        t = r["deciding_target"] if deciding else r["target"]
        g = game(0, 0, p, t, r["cap"])
        return g * from_sets(sa + 1, sb) + (1 - g) * from_sets(sa, sb + 1)

    if sets_a >= need or sets_b >= need:
        return 1.0 if sets_a > sets_b else 0.0
    deciding = sets_a == sets_b == need - 1
    t = r["deciding_target"] if deciding else r["target"]
    g = game(int(a), int(b), p, t, r["cap"])
    return g * from_sets(sets_a + 1, sets_b) + (1 - g) * from_sets(sets_a, sets_b + 1)


def set_score_chances(p, sport):
    """Chance of each final set score, e.g. volleyball {'3-0': .., '3-1': .., ..., '0-3': ..}."""
    r = RULES[sport]
    need = r["best_of"] // 2 + 1
    out = {}

    def walk(sa, sb, prob):
        if sa == need or sb == need:
            key = f"{sa}-{sb}"
            out[key] = out.get(key, 0) + prob
            return
        deciding = sa == sb == need - 1
        g = game(0, 0, float(round(p, 4)), r["deciding_target"] if deciding else r["target"], r["cap"])
        walk(sa + 1, sb, prob * g)
        walk(sa, sb + 1, prob * (1 - g))
    walk(0, 0, 1.0)
    return out


def point_value(p, target, cap=None, a=0, b=0):
    """How much one rally swings the game: P(win | win this rally) - P(win | lose it)."""
    return game(a + 1, b, p, target, cap) - game(a, b + 1, p, target, cap)


def update_p(p0, won, played, strength):
    """Blend the pre-match estimate with what has happened so far (a Bayesian beta update).
    strength = how many rallies the pre-match estimate is 'worth'."""
    return (p0 * strength + won) / (strength + played)


def logit(x):
    x = np.clip(x, 1e-9, 1 - 1e-9)
    return np.log(x / (1 - x))


def expit(z):
    return 1 / (1 + np.exp(-z))
