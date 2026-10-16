"""Validate capped-prefix traces while preserving the frozen v1 summarizer."""

import argparse
import hashlib
from pathlib import Path
import json

if __package__:
    from . import summarize_approach_trace as original_summary
    from .panda_capped_action_prefix import CappedActionPrefixSchedule
else:
    import summarize_approach_trace as original_summary
    from panda_capped_action_prefix import CappedActionPrefixSchedule


def summarize(path):
    # CLI/single-threaded adapter; restore the original factory even on errors.
    factory = original_summary.ActionPrefixSchedule
    original_summary.ActionPrefixSchedule = CappedActionPrefixSchedule
    try:
        result = original_summary.summarize(path)
    finally:
        original_summary.ActionPrefixSchedule = factory
    result['summary_driver_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    helper = Path(__file__).with_name('panda_capped_action_prefix.py')
    result['capped_prefix_source_sha256'] = hashlib.sha256(helper.read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    result = summarize(args.trace)
    with args.output.open('x') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f"Complete episodes: {result['complete_episodes']}; successes: {result['successes']}")
    for episode in result['episodes']:
        print('episode', episode['episode_index'], 'Y', episode['position']['y'],
              'steps', episode['steps'], 'descent', episode['first_executed_descent_step'],
              'close', episode['first_executed_close_step'])
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
