"""QA a fixed-home 2D X/Y corner-position dataset."""

from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

HOME_X = 0.40
EXPECTED = ((0.32, -0.075), (0.32, 0.075), (0.48, -0.075), (0.48, 0.075))

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--episodes-per-position", type=int, default=5)
    p.add_argument("--min-active-frames", type=int, default=4)
    args = p.parse_args()
    root=args.dataset_root
    schedule=json.loads((root/'position_schedule.json').read_text())
    history=json.loads((root/'position_history.json').read_text())
    info=json.loads((root/'meta/info.json').read_text())
    files=sorted(root.glob('data/chunk-*/file-*.parquet'))
    if len(schedule['positions']) != len(history): raise ValueError('schedule/history mismatch')
    data=pq.read_table(files,columns=['action','episode_index','next.reward','next.done']).to_pandas()
    actions=np.stack(data['action'].to_numpy()).astype(np.float32); eps=data['episode_index'].to_numpy(dtype=np.int64)
    positions=[(round(float(x['x']),6),round(float(x['y']),6)) for x in schedule['positions']]
    counts=Counter(positions); errors=[]
    print('Dataset root:',root); print('Split:',schedule.get('split')); print('FPS:',info.get('fps')); print('Episodes:',data['episode_index'].nunique()); print('Frames:',len(data)); print('Position counts:',dict(sorted(counts.items()))); print('Reward-positive frames:',int((data['next.reward']>0).sum())); print('Done frames:',int(data['next.done'].sum())); print()
    if sorted(counts) != sorted(EXPECTED): errors.append(f'expected positions {sorted(EXPECTED)}, found {sorted(counts)}')
    for pos in EXPECTED:
        if counts[pos] != args.episodes_per_position: errors.append(f'{pos} has {counts[pos]} episodes')
    active_x_total=active_y_total=correct_x=correct_y=overlap=0; failed=[]
    for ep,pos in enumerate(positions):
        mask=eps==ep; a=actions[mask]; ax=np.abs(a[:,0])>1e-6; ay=np.abs(a[:,1])>1e-6
        sx=np.sign(pos[0]-HOME_X); sy=np.sign(pos[1]); cx=ax&(np.sign(a[:,0])==sx); cy=ay&(np.sign(a[:,1])==sy)
        rx=int((a[:,3]>=1.5).sum()); reward=float(data.loc[mask,'next.reward'].sum()); done=int(data.loc[mask,'next.done'].sum())
        if reward<=0 or done==0: failed.append(ep)
        active_x_total+=int(ax.sum()); active_y_total+=int(ay.sum()); correct_x+=int(cx.sum()); correct_y+=int(cy.sum()); overlap+=int((ax&ay).sum())
        print(f'ep={ep:02d} pos=({pos[0]:.3f},{pos[1]:+.3f}) frames={int(mask.sum()):3d} x_active={int(ax.sum()):2d} x_sign={int(cx.sum())/max(int(ax.sum()),1):.0%} y_active={int(ay.sum()):2d} y_sign={int(cy.sum())/max(int(ay.sum()),1):.0%} xy_overlap={int((ax&ay).sum()):2d} close={rx:2d} reward={reward:.1f} done={done}')
        if int(ax.sum())<args.min_active_frames or int(ay.sum())<args.min_active_frames: errors.append(f'episode {ep} has insufficient X/Y active frames')
        if ax.any() and int(cx.sum())/int(ax.sum())<.8: errors.append(f'episode {ep} X sign agreement <80%')
        if ay.any() and int(cy.sum())/int(ay.sum())<.8: errors.append(f'episode {ep} Y sign agreement <80%')
    if failed: errors.append(f'failed demonstrations: {failed}')
    print('\nTotal active X frames:',active_x_total); print('Overall X sign agreement:',f'{correct_x/max(active_x_total,1):.1%}'); print('Total active Y frames:',active_y_total); print('Overall Y sign agreement:',f'{correct_y/max(active_y_total,1):.1%}'); print('Frames with simultaneous X/Y activity:',overlap); print('Failed demonstrations:',failed); print('Manual video check required: confirm arm visibility and diagonal approach.')
    if errors:
        print('\nQA: FAIL'); [print('-',e) for e in errors]; return 1
    print('\nQA: PASS'); return 0

if __name__ == '__main__': raise SystemExit(main())
