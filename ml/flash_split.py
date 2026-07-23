"""
Deterministic train/eval split for confirmed flash photos, shared by
train_clip.py and eval_recognition.py.

Every flash_id lands in the same bucket forever (no state file to keep in
sync) -- 1-in-EVAL_HOLDOUT_MOD is held out for eval and never trained on, so
eval_recognition.py measures generalization instead of memorization.
"""

EVAL_HOLDOUT_MOD = 5  # 1 in 5 confirmed photos (~20%) held out for eval

def is_eval(flash_id):
    return int(flash_id) % EVAL_HOLDOUT_MOD == 0
