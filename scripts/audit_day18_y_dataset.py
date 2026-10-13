"""QA Day18's new shorter-Y recordings before appending them to Day17.

Expected X={0.36,0.44}, Y={-0.0375,+0.0375}; one episode per position
for smoke, five for the full new recording. Uses the unchanged phase-order,
sign, hover, close, and lift requirements. This audits ONLY the new recording,
not the eventual 80-episode mixture. Visual inspection must confirm stopping
above the actual cube; action label checks alone cannot establish alignment.
"""

if __package__:
    from .audit_day16_xy_dataset import main as audit
else:
    from audit_day16_xy_dataset import main as audit

ADDED_POSITIONS = tuple(
    (x, y) for x in (0.36, 0.44) for y in (-0.0375, 0.0375)
)


def main(argv=None):
    return audit(argv, expected_positions=ADDED_POSITIONS, description=__doc__)


if __name__ == '__main__':
    raise SystemExit(main())
