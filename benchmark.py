"""Paired trials with a fixed one-frame policy and one extra camera position.

Run: python benchmark.py [--split dev|test] [--scene T000] [--out PATH]
"""
import argparse
import csv
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from src.docking import validate
from src.geometry import estimate
from src.perception import detect
from src.scene import Pallet, ground_truth, render_scene, solid_boxes


ROOT = Path(__file__).resolve().parent
DISTANCES = (1.8, 2.15, 2.5)
YAWS = (-16, -8, 0, 8, 16)
VISIBILITIES = ("none", "partial", "severe")
REVISIT_Y = -0.50
SIGMA_M = 0.0015
DROPOUT = 0.001


@dataclass(frozen=True)
class Trial:
    id: str
    seed: int
    layout: int
    distance: float
    yaw: float
    occlusion: str
    pallets: tuple
    kind: str = "single"


def design(split):
    test = split == "test"
    base = 745301 if test else 2309
    prefix = "T" if test else "D"
    trials = []
    for layout in range(4 if test else 1):
        for distance in DISTANCES:
            for yaw in YAWS:
                for occ in VISIBILITIES:
                    i = len(trials)
                    seed = base + i
                    rng = np.random.default_rng(seed)
                    pallet = Pallet(x=distance+rng.uniform(-.025, .025), y=rng.uniform(-.09,.09),
                                    yaw_deg=yaw+rng.uniform(-1.,1.))
                    trials.append(Trial(f"{prefix}{i:03}", seed, layout, distance, yaw, occ, (pallet,)))
    for i in range(4):
        angle = (-10., -5., 5., 10.)[i]
        pallets = (Pallet(x=2.6, y=.62, yaw_deg=angle), Pallet(x=2.7, y=-.62, yaw_deg=-angle))
        trials.append(Trial(f"{prefix}N{i}", base+900+i, i, 0., 0., "none", (), "empty"))
        trials.append(Trial(f"{prefix}M{i}", base+920+i, i, 2.6, angle, "none", pallets, "ambiguous"))
    return trials


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_default, allow_nan=False), encoding="utf8")


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantiles(values):
    if not values:
        return None
    return dict(zip(("median", "p95"), map(float, np.percentile(values, (50,95)))))


def rate(n, d):
    return n / d if d else None


def observe(trial, attempt):
    start = perf_counter()
    y = 0. if attempt == 0 else REVISIT_Y
    rgb, depth, k, transform = render_scene(trial.pallets, view_y=y, occlusion=trial.occlusion)
    rng = np.random.default_rng(np.random.SeedSequence([trial.seed, attempt, 303]))
    valid = np.isfinite(depth) & (depth > 0)
    depth = depth.copy()
    depth[valid] += rng.normal(0., SIGMA_M, valid.sum()).astype(depth.dtype)
    depth[rng.random(depth.shape) < DROPOUT] = 0.
    after_render = perf_counter()
    candidates = detect(rgb, depth, k, transform)
    end = perf_counter()
    return {"rgb":rgb, "depth":depth, "K":k, "T":transform, "candidates":candidates, "render_ms":(after_render-start)*1000,
            "perception_ms":(end-after_render)*1000}


def ideal_marks(frame, truth):
    """Evaluator-only oracle clicks. Hidden face samples are dropped by truth depth."""
    if truth is None:
        return None, "NO_UNIQUE_TARGET"
    n=np.asarray(truth["inward_world"])
    left=np.array([-n[1],n[0],0.])
    mid=truth["center_world_m"]
    surface=[mid+y*left+[0.,0.,z] for y,z in
             ((-.34,.124),(.33,.125),(-.27,-.125),(.29,-.126),(0.,0.),(.45,0.),(-.45,0.))]
    t,k,depth=frame["T"],frame["K"],frame["depth"]
    def project(points):
        camera=(np.asarray(points)-t[:3,3])@t[:3,:3]
        uvz=camera@k.T
        return np.rint(uvz[:,:2]/uvz[:,2:]).astype(int),camera[:,2]
    marks,z=project(surface)
    def visible(uv):
        return bool(0<=uv[0]<depth.shape[1] and 0<=uv[1]<depth.shape[0])
    marks=[p for p,true_z in zip(marks,z) if visible(p) and abs(depth[p[1],p[0]]-true_z)<.01]
    holes,_=project(truth["holes_world_m"])
    if len(marks)<3 or not all(visible(p) for p in holes):
        return None,"INSUFFICIENT_ANNOTATIONS"
    try:
        result=estimate(depth,k,t,marks,holes)
    except ValueError:
        return None,"DEGENERATE_MARKS"
    return {"midpoint_world":result["midpoint_world"],"yaw_deg":result["yaw_deg"]},"ORACLE_PIXELS"


def decide(candidates):
    # This benchmark's command is "dock the only identifiable pallet".
    if not candidates:
        return None, "NO_CANDIDATE"
    if len(candidates) > 1:
        if all(c["status"] != "ACCEPT" for c in candidates):
            return None, "FRAGMENTED_OR_OCCLUDED"
        return None, "AMBIGUOUS"
    target = candidates[0]
    if target["status"] != "ACCEPT":
        return None, target["reason"]
    return {"midpoint_world":target["center_world_m"], "yaw_deg":target["yaw_deg"]}, "ACCEPT"


def evaluate(trial, cmd, reason, truth, boxes, oracle_ok, frames, strategy):
    accepted = cmd is not None
    success = False
    e_pos = e_yaw = ex = ey = ez = None
    dock_ms = None
    dock_status = "not_attempted"
    if accepted and trial.kind == "single":
        start = perf_counter()
        verdict = validate(truth, cmd, obstacles=boxes)
        dock_ms = (perf_counter()-start)*1000
        success = bool(verdict["success"])
        dock_status = verdict["status"]
        d = np.asarray(cmd["midpoint_world"])-truth["center_world_m"]
        e_pos = float(1000*np.linalg.norm(d))
        yaw = np.deg2rad(truth["yaw_deg"])
        ex = float(1000*(np.cos(yaw)*d[0]+np.sin(yaw)*d[1]))
        ey = float(1000*(-np.sin(yaw)*d[0]+np.cos(yaw)*d[1]))
        ez = float(1000*d[2])
        e_yaw = abs((cmd["yaw_deg"]-truth["yaw_deg"]+180.) % 360.-180.)
    elif accepted:
        dock_status = "wrong_target"
    return dict(scene_id=trial.id, seed=trial.seed, layout=trial.layout, distance_bin_m=trial.distance,
                yaw_bin_deg=trial.yaw, visibility=trial.occlusion, kind=trial.kind,
                strategy=strategy, observations=len(frames), accepted=accepted, success=success,
                oracle_feasible=oracle_ok, reason=reason, dock_status=dock_status,
                position_error_mm=e_pos, local_forward_error_mm=ex, local_left_error_mm=ey,
                local_up_error_mm=ez, yaw_error_deg=e_yaw,
                render_ms=sum(f["render_ms"] for f in frames) if frames else None,
                perception_ms=sum(f["perception_ms"] for f in frames) if frames else None,
                dock_ms=dock_ms)


def save_replay(folder, trial, frames, rows):
    folder.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        image=cv2.cvtColor(frame["rgb"],cv2.COLOR_RGB2BGR)
        for c in frame["candidates"]:
            x0,y0,x1,y1 = map(int,c["bbox_xyxy"])
            color = (68,170,63) if c["status"]=="ACCEPT" else (30,80,235)
            cv2.rectangle(image,(x0,y0),(x1,y1),color,2)
            cv2.putText(image,f"{c['id']} {c['status']}",(max(x0,1),max(y0-6,13)),
                        cv2.FONT_HERSHEY_SIMPLEX,.48,color,1,cv2.LINE_AA)
        ok,data = cv2.imencode(".png",image)
        if not ok:
            raise ValueError("Could not encode replay")
        (folder/("first.png" if i==0 else "revisit.png")).write_bytes(data.tobytes())
    write_json(folder/"record.json", {"trial":trial.id,"outcomes":rows,
        "candidates":[f["candidates"] for f in frames]})


def summarize(rows):
    n=len(rows)
    accepted=[r for r in rows if r["accepted"]]
    feasible=[r for r in rows if r["oracle_feasible"]]
    successes=sum(r["success"] for r in rows)
    errors=[r for r in accepted if r["kind"]=="single"]
    return dict(tasks=n,accepted=len(accepted),refused=n-len(accepted),success=successes,
                false_accepts=sum(not r["success"] for r in accepted),
                oracle_feasible=len(feasible),coverage=rate(len(accepted),n),
                wrong_release_rate=rate(sum(not r["success"] for r in accepted),len(accepted)),
                feasible_success_rate=rate(sum(r["success"] for r in feasible),len(feasible)),
                feasible_rejections=sum(not r["accepted"] for r in feasible),
                revisits=sum(r["observations"]==2 for r in rows),
                position_error_mm=quantiles([r["position_error_mm"] for r in errors]),
                yaw_error_deg=quantiles([r["yaw_error_deg"] for r in errors]),
                perception_ms=quantiles([r["perception_ms"] for r in rows if r["perception_ms"] is not None]),
                rendering_ms=quantiles([r["render_ms"] for r in rows if r["render_ms"] is not None]),
                validator_ms=quantiles([r["dock_ms"] for r in rows if r["dock_ms"] is not None]))


def run(trials, out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    saved = {}
    for j, trial in enumerate(trials):
        boxes = solid_boxes(trial.pallets,occlusion=trial.occlusion)
        truth = ground_truth(trial.pallets[0]) if trial.kind=="single" else None
        oracle = {"midpoint_world":truth["center_world_m"],"yaw_deg":truth["yaw_deg"]} if truth else None
        oracle_ok = bool(oracle and validate(truth,oracle,obstacles=boxes)["success"])
        oracle_row=evaluate(trial,oracle,"TRUTH_POSE" if truth else "NO_UNIQUE_TARGET",
                            truth,boxes,oracle_ok,[],"oracle")
        frames=[observe(trial,0)]
        marked_cmd,marked_reason=ideal_marks(frames[0],truth)
        marked=evaluate(trial,marked_cmd,marked_reason,truth,boxes,oracle_ok,[],"ideal_marks")
        marked["observations"]=1
        cmd,reason=decide(frames[0]["candidates"])
        fixed=evaluate(trial,cmd,reason,truth,boxes,oracle_ok,frames,"single_view")
        if cmd is None and reason!="AMBIGUOUS":
            frames.append(observe(trial,1))
            cmd,reason=decide(frames[-1]["candidates"])
        active=evaluate(trial,cmd,reason,truth,boxes,oracle_ok,frames,"one_revisit")
        trio=[oracle_row,marked,fixed,active]
        rows.extend(trio)
        if trial.kind!="single":
            category=trial.kind
        elif active["accepted"] and not active["success"]:
            category="false_accept"
        elif active["success"] and not fixed["success"]:
            category="recovered"
        elif not active["accepted"]:
            category="refused"
        else:
            category="easy"
        if category not in saved:
            saved[category]=trial.id
            save_replay(out/"replays"/category,trial,frames,trio)
        if (j+1)%30==0:
            print(f"processed {j+1}/{len(trials)}",flush=True)
    policies=("oracle","ideal_marks","single_view","one_revisit")
    single=[r for r in rows if r["kind"]=="single"]
    by_policy={p:summarize([r for r in single if r["strategy"]==p]) for p in policies}
    controls=[r for r in rows if r["kind"]!="single"]
    by_control={k:{p:summarize([r for r in controls if r["strategy"]==p and r["kind"]==k])
                       for p in policies} for k in ("empty","ambiguous")}
    groups=[]
    for selector, levels in (("visibility",VISIBILITIES),("distance_bin_m",DISTANCES),("yaw_bin_deg",YAWS)):
        for value in levels:
            for policy in policies:
                group=[r for r in single if r[selector]==value and r["strategy"]==policy]
                if group:
                    metric=summarize(group)
                    groups.append({"group":selector,"value":value,"strategy":policy,
                        **{k:v for k,v in metric.items() if k not in ("position_error_mm","yaw_error_deg","perception_ms","rendering_ms","validator_ms")}})
    write_csv(out/"trials.csv",rows)
    write_csv(out/"by_visibility.csv", [g for g in groups if g["group"]=="visibility"])
    write_csv(out/"groups.csv",groups)
    write_json(out/"summary.json",{"single_pallet":by_policy,"controls":by_control,"replays":saved,
        "configuration":{"sigma_axial_m":SIGMA_M,"dropout_fraction":DROPOUT,"second_camera_y_m":REVISIT_Y,
            "python":platform.python_version(),"numpy":np.__version__,"opencv":cv2.__version__,
            "opencv_threads":cv2.getNumThreads(),"timing":"CPU wall clock, warmed up; render includes noise. Re-observation includes both renders / detector calls. Excludes physical transit."}})
    print(json.dumps({k:{m:v[m] for m in ("tasks","success","accepted","false_accepts","revisits")} for k,v in by_policy.items()}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev","test"),default="test")
    parser.add_argument("--scene",help="Reproduce one scene by ID")
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    trials=design(args.split)
    if args.scene:
        trials=[t for t in trials if t.id==args.scene]
        if not trials:
            parser.error("ID not present in this split")
    folder=args.out or ROOT/"results"/("benchmark" if args.split=="test" else "development")
    if args.scene and args.out is None:
        folder=ROOT/"results"/"reproduced"/args.scene
    observe(design("dev")[0],0)
    run(trials,folder)
