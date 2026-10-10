"""QA Day17's preserved-endpoint mixture: 10 per endpoint, 5 per intermediate.

Uses the existing phase-order checks without relaxing any Day16 thresholds.
Manual camera review is still required; automatic merge/video verification is
recorded separately in dataset_provenance.json by build_day17_dataset.py.
"""

if __package__:
    from .audit_day16_xy_dataset import EXPECTED_POSITIONS, main as audit
else:
    from audit_day16_xy_dataset import EXPECTED_POSITIONS, main as audit


def main(argv=None):
    counts = {p: 10 if p[0] in (0.32, 0.48) else 5 for p in EXPECTED_POSITIONS}
    return audit(argv, expected_counts=counts, description=__doc__)


if __name__ == '__main__':
    raise SystemExit(main())
