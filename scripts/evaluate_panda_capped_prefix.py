"""Run the existing evaluator with an explicit capped initial-prefix scheduler.

Only the scheduler class binding in this process is replaced. No frozen source
file is edited, and the original ACT/action/environment paths are reused.
Invoke as: python scripts/evaluate_panda_capped_prefix.py [evaluator flags].
"""

import evaluate_panda_act as evaluator
from panda_capped_action_prefix import CappedActionPrefixSchedule


def main():
    original = evaluator.ActionPrefixSchedule
    evaluator.ActionPrefixSchedule = CappedActionPrefixSchedule
    try:
        print('Capped-prefix diagnostic: shorten the final initial prefix to keep the switch step exact.')
        evaluator.main()
    finally:
        evaluator.ActionPrefixSchedule = original


if __name__ == '__main__':
    main()
