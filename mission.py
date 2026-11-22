"""Run a language-guided pick through visual grounding, pose estimation and a virtual insertion."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.docking import validate
from src.perception import detect
from src.scene import Pallet, ground_truth, render_scene, solid_boxes
from src.semantic import Grounder


ROOT = Path(__file__).resolve().parent


def decide(rgb, depth, K, T, selector, instruction):
    """Sensor-side decision; receives no pallet objects, labels, or target pose."""
    candidates = detect(rgb, depth, K, T)
    # A partially observed alternative could also match the instruction.
    if any(c['status'] != 'ACCEPT' for c in candidates):
        return 'REOBSERVE', None, candidates
    answer = selector.select(rgb, candidates, instruction)['decision']
    selected = next((c for c in candidates if c['id'] == answer), None)
    return answer, selected, candidates


def run_task(pallets, visibility, instruction, expected, selector, folder):
    """The expected instance lives exclusively in this simulator/evaluator."""
    events = []
    folder.mkdir(parents=True, exist_ok=True)
    contact = None
    selected = None
    for index, camera_y in enumerate((0., -.5)):
        rgb, depth, K, T = render_scene(pallets, camera_y, visibility)
        decision, selected, candidates = decide(rgb, depth, K, T, selector, instruction)
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for c in candidates:
            x0,y0,x1,y1 = map(int,c['bbox_xyxy'])
            color = (40,220,125) if c is selected else (20,160,235)
            cv2.rectangle(frame, (x0,y0), (x1,y1), color, 2)
            cv2.putText(frame,c['id']+' '+c['status'],(max(0,x0),max(20,y0-6)),0,.48,color,1,cv2.LINE_AA)
        cv2.imencode('.png', frame)[1].tofile(folder/f'view-{index}.png')
        events.append({'view_y_m':camera_y, 'decision':decision,
                       'candidates':[{'id':c['id'],'geometry':c['status'],'reason':c['reason']} for c in candidates]})
        if decision != 'REOBSERVE':
            break
    # A reference target is never chosen by passing the expected label to perception.
    if selected is not None:
        if isinstance(expected,int):
            contact = validate(ground_truth(pallets[expected]), selected, solid_boxes(pallets,visibility))
            outcome = 'DOCKED' if contact['success'] else 'FAILED_DOCK'
        else:
            outcome = 'WRONG_RELEASE'
    else:
        outcome = 'INSUFFICIENT_VIEW' if decision == 'REOBSERVE' else decision
    wanted = 'DOCKED' if isinstance(expected,int) else expected
    row = {'instruction':instruction,'observations':events,'outcome':outcome,'expected':wanted,
           'success':outcome == wanted,'geometry_verdict':contact}
    (folder/'record.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
    return row


def tasks():
    left = Pallet(2.8,.65,-8.,'blue')
    right = Pallet(2.8,-.65,9.,'red')
    both_blue = (left, Pallet(2.8,-.65,9.,'blue'))
    request = '取蓝色料箱下方的托盘'
    return [
        ('blue-left', (left,right), 'none',request,0),
        ('red-right', (left,right),'none','取红色料箱下方的托盘',1),
        ('relative', both_blue,'none','取右侧蓝色料箱下方的托盘',1),
        ('absent',both_blue,'none','取红色料箱下方的托盘','ABSENT'),
        ('ambiguous',both_blue,'none',request,'AMBIGUOUS'),
        ('unsupported',(left,right),'none','把易碎货物送到三号站','UNSUPPORTED'),
        ('reobserve',(Pallet(2.,0.,12.,'blue'),),'partial',request,0),
        ('occluded',(Pallet(2.,0.,12.,'blue'),),'severe',request,'INSUFFICIENT_VIEW'),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'results'/'missions')
    args = parser.parse_args()
    selector = Grounder.load()
    rows = {}
    for name, pallets, visibility, command, expected in tasks():
        row = run_task(pallets,visibility,command,expected,selector,args.out/name)
        rows[name] = row
        print(f"{name:14} {row['outcome']:20} {'PASS' if row['success'] else 'FAIL'}")
    (args.out/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    return 0 if all(row['success'] for row in rows.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
