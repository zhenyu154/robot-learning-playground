"""QA the full 80-episode Day18 mixture; this is not the 20-episode source audit.

Preserve all 60 Day17 episodes (10 per endpoint, 5 per intermediate X at
Y=±0.075), plus 5 per added combination at X={0.36,0.44}, Y=±0.0375.
Uses unchanged phase-order thresholds. Review the source/merged camera views.
"""

if __package__:
    from .audit_day16_xy_dataset import EXPECTED_POSITIONS, main as audit
    from .audit_day18_y_dataset import ADDED_POSITIONS
else:
    from audit_day16_xy_dataset import EXPECTED_POSITIONS, main as audit
    from audit_day18_y_dataset import ADDED_POSITIONS

EXPECTED_COUNTS = {
    **{p: 10 if p[0] in (0.32, 0.48) else 5 for p in EXPECTED_POSITIONS},
    **{p: 5 for p in ADDED_POSITIONS},
}


def main(argv=None):
    return audit(argv, expected_counts=EXPECTED_COUNTS, description=__doc__)


if __name__ == '__main__':
    raise SystemExit(main())
