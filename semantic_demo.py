"""Train and evaluate the closed-vocabulary colour/side grounding experiment."""
import argparse
import json
import os
from pathlib import Path

# Small matrices are faster on one CPU thread, and this makes replay consistent.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import cv2
import numpy as np

from src.scene import Pallet, render_scene
from src.semantic import DEFAULT_WEIGHTS, Grounder, encode_text, image_features, mlp, select_by_rule


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "semantic"
# Stratified colour layouts include counterexamples in which the colour is absent
# and same-colour doubles in which a unique target cannot be determined.
LAYOUTS = [("blue","red"), ("red","blue"), ("red","red"), ("blue","blue"),
           ("red",None), (None,"blue"), (None,None), ("blue",None), (None,"red")]


def requests(test):
    queries = []
    for side in ("", "左", "右"):
        for name, char in ((None,""), ("blue","蓝"), ("red","红")):
            direction = side + ("边" if test else "侧") if side else ""
            load = char + ("色箱子下面的" if test else "色料箱下方的") if char else ""
            queries.append((("请选" if test else "取") + direction + load + "托盘", side, name))
    return queries


def targets(side, cargo, colours):
    return [((not side or side == ("左","右")[i]) and (cargo is None or colour == cargo))
            for i,colour in enumerate(colours)]


def expected(matches):
    ids = [f"P{i+1}" for i,match in enumerate(matches) if match]
    return ids[0] if len(ids) == 1 else ("ABSENT" if not ids else "AMBIGUOUS")


def project_bounds(p, K, T):
    # Oracle boxes are permitted only for training supervised candidate crops and
    # verifying test proposals. Forward inference receives detector candidates.
    angle = np.deg2rad(p.yaw_deg)
    c, s = np.cos(angle),np.sin(angle)
    world = np.array([[p.x+dx*c-dy*s, p.y+dx*s+dy*c, z]
                      for dx in (0,.40) for dy in (-.5,.5) for z in (0,.40)])
    xyz = (world - T[:3,3]) @ T[:3,:3]
    uvw = xyz @ K.T
    uv = uvw[:,:2] / uvw[:,2:]
    return np.r_[uv.min(axis=0),uv.max(axis=0)].astype(int).tolist()


def scene_sample(seed, layout):
    rng = np.random.default_rng(seed)
    # Lateral separation keeps both bodies visible. Distances/yaw/elevation are
    # independent of colour and vary across seed-separated training/test layouts.
    x = rng.uniform(2.94, 3.27)
    pallets = [Pallet(x=float(x+rng.uniform(-.09,.09)),y=float(sign*rng.uniform(.63,.69)),
                      yaw_deg=float(rng.uniform(-11,11)),cargo=colour)
               for sign,colour in zip((1,-1), layout)]
    rgb,depth,K,T = render_scene(pallets,view_y=float(rng.uniform(-.09,.09)))
    return rgb,depth,K,T,[{"id":f"P{i+1}","bbox_xyxy":project_bounds(p,K,T)}
                          for i,p in enumerate(pallets)]


def make_training_features():
    rng = np.random.default_rng(1310)
    image_rows, texts, labels = [],[],[]
    for i in range(27):
        layout = LAYOUTS[i % len(LAYOUTS)]
        rgb,_,_,_,boxes = scene_sample(11000+i, layout)
        for _ in range(7):
            # Calibration and colour jitter keep the encoder from memorising a
            # single RGB triplet or crop. No test observation enters training.
            shift = rng.integers(-5,6,(2,4))
            distorted = [{"bbox_xyxy":np.asarray(c["bbox_xyxy"])+d, "id":c["id"]}
                         for c,d in zip(boxes,shift)]
            _, features = image_features(rgb,distorted)
            grid = features[:,:-1].reshape(2,4,4,3)
            grid[:] = np.clip(grid*rng.uniform(.85,1.15,(1,1,1,3))+
                              rng.uniform(-.10,.10,(1,1,1,3)),-1,1)
            for query,side,colour in requests(False):
                image_rows.extend(features)
                texts.extend([encode_text(query)]*2)
                labels.extend(targets(side, colour, layout))
    return tuple(np.array(x,dtype=np.float32) for x in (image_rows,texts,labels))


def train():
    X,T,Y = make_training_features()
    rng = np.random.default_rng(2026)
    def weight(n,m):
        return (rng.normal(size=(n,m))*np.sqrt(1/n)).astype(np.float32)
    n = 20
    P = {"Vi":weight(X.shape[1],n), "bi":np.zeros(n,"f"), "Wt":weight(T.shape[1],n),
         "bt":np.zeros(n,"f"), "Wh":weight(3*n,n), "bh":np.zeros(n,"f"),
         "Wo":weight(n,1), "bo":np.zeros(1,"f")}
    momentum = {k:np.zeros_like(v) for k,v in P.items()}
    variance = {k:np.zeros_like(v) for k,v in P.items()}
    for step in range(1,1201):
        inds = rng.integers(0,len(Y),128)
        x,t,y = X[inds],T[inds],Y[inds]
        prob, (vi,tx,f,h) = mlp(P,x,t)
        dz = ((prob-y)/len(y))[:,None]
        dh = (dz @ P["Wo"].T)*(1-h*h)
        dv,dt,dp = np.split(dh @ P["Wh"].T,3,axis=1)
        dv = (dv+dp*tx)*(1-vi*vi)
        dt = (dt+dp*vi)*(1-tx*tx)
        grads = {"Wo":h.T @ dz,"bo":dz.sum(0),"Wh":f.T @ dh,"bh":dh.sum(0),
                 "Vi":x.T @ dv,"bi":dv.sum(0),"Wt":t.T @ dt,"bt":dt.sum(0)}
        for k,g in grads.items():
            momentum[k] = .9*momentum[k]+.1*g
            variance[k] = .999*variance[k]+.001*g*g
            P[k] -= .004*(momentum[k]/(1-.9**step))/(np.sqrt(variance[k]/(1-.999**step))+1e-8)
    DEFAULT_WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(DEFAULT_WEIGHTS,**P)
    print(f"Saved {sum(v.size for v in P.values())} parameters; {len(Y)} labelled candidate pairs")


def iou(a,b):
    a,b = np.array(a),np.array(b)
    area = np.prod(np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])))
    return area/(np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-area)


def showcase(rgb, candidates, predictions):
    image = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for c in candidates:
        x,y,x1,y1 = c["bbox_xyxy"]
        cv2.rectangle(image,(x,y),(x1,y1),(10,235,156),2)
        cv2.putText(image,c["id"],(x+5,y+18),0,.6,(10,235,156),2,cv2.LINE_AA)
    panel = np.full((111,image.shape[1],3), (43,33,25),np.uint8)
    for i,(label,value) in enumerate(predictions):
        cv2.putText(panel,f"{label}  =>  {value}",(12,24+i*25),0,.55,(246,246,246),1,cv2.LINE_AA)
    cv2.imencode(".png",np.vstack((image,panel)))[1].tofile(OUTPUT/"selection.png")


def evaluate():
    from src.perception import detect
    model = Grounder.load()
    records = []
    for i in range(18):
        seed = 39000 + i
        layout = LAYOUTS[i % len(LAYOUTS)]
        rgb,depth,K,T,oracle = scene_sample(seed,layout)
        candidates = detect(rgb,depth,K,T)
        # Remove chroma from the SAME RGB observation, preserving luminance,
        # arrangement, generated IDs, depth and candidate geometry.
        gray = cv2.cvtColor(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        ordered = sorted(candidates,key=lambda c:c["bbox_xyxy"][0]+c["bbox_xyxy"][2])
        aligned = (len(ordered)==2 and all(c["id"]==ref["id"] and
                   iou(c["bbox_xyxy"],ref["bbox_xyxy"])>.35 for c,ref in zip(ordered,oracle)))
        predictions = []
        for query,side,colour in requests(True):
            truth = expected(targets(side,colour,layout))
            result = model.select(rgb,candidates,query)
            baseline = select_by_rule(rgb,candidates,query)
            no_colour = model.select(gray,candidates,query)
            no_instruction = model.select(rgb,candidates,"取托盘")
            records.append({"seed":seed,"instruction":query,"cargo_left_to_right":layout,
                            "expected":truth, "decision":result["decision"],"scores":result["scores"],
                            "rule":baseline["decision"], "greyscale":no_colour["decision"],
                            "no_instruction":no_instruction["decision"],"candidates_aligned":bool(aligned),
                            "selector_latency_ms":result["latency_ms"]})
            if not side or (side=="右" and colour=="blue"):
                label = ({"blue":"below blue cargo","red":"below red cargo",
                          None:"any pallet"}[colour])
                predictions.append((label+(" on the right" if side else ""),result["decision"]))
        if i==0:
            showcase(rgb,candidates,predictions)
    valid = [r for r in records if r["candidates_aligned"]]
    def correct(key,subset):
        return sum(r["candidates_aligned"] and r[key]==r["expected"] for r in subset)
    stats = {"scope":"two rendered pallet bodies; candidate proposal + semantic choice, before docking",
             "train_scenes":27, "test_scenes":18,"test_tasks":len(records),
             "aligned_proposal_tasks":len(valid),"grounder_correct":correct("decision",records),
             "rule_correct":correct("rule",records),
             "greyscale_correct":correct("greyscale",records),
             "no_instruction_correct":correct("no_instruction",records),
             "aligned_grounder_correct":correct("decision",valid),
             "by_expected":{k:{"count":len(sub),"correct":correct("decision",sub)}
                            for k in ("P1","P2","ABSENT","AMBIGUOUS")
                            for sub in [[r for r in records if r["expected"]==k]]},
             "selector_latency_p50_ms":float(np.median([r["selector_latency_ms"] for r in records])),
             "weights_parameters":sum(arr.size for arr in model.params.values())}
    for file,obj in (("metrics.json",stats),("predictions.json",records)):
        (OUTPUT/file).write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(stats,indent=2))


def qwen_example(model_directory):
    from src.perception import detect
    from src.semantic_qwen import LocalQwenSelector

    rgb,depth,K,T,_ = scene_sample(39000,LAYOUTS[0])
    candidates = detect(rgb,depth,K,T)
    selector = LocalQwenSelector(model_directory)
    rows = []
    for query,side,colour in requests(True):
        rows.append({"instruction":query,"expected":expected(targets(side,colour,LAYOUTS[0])),
                     **selector.select(rgb,candidates,query)})
    dest = ROOT/"results"/"semantic_qwen"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"example.json").write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding="utf-8")
    print(f"{len(rows)} Qwen predictions saved to {dest}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train",action="store_true",help="train and overwrite the local grounding model first")
    mode.add_argument("--qwen-dir",type=Path,help="run a separate example using local Qwen2-VL-2B-Instruct weights")
    options = parser.parse_args()
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if options.qwen_dir:
        qwen_example(options.qwen_dir)
    else:
        if options.train:
            train()
        evaluate()
