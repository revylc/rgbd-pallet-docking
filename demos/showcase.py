"""Build the offline project viewer from rendered and measured project outputs."""
import base64
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.perception import detect
from src.scene import Pallet, render_scene
from src.docking import tine_corners, START_GAP_M
from src.scene import solid_boxes
from src.semantic import Grounder
from mission import decide, tasks
OUT = ROOT / "results" / "showcase"
FRONT = ROOT / "sample_run" / "front"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def rgbd_cloud():
    """Unproject observed surface samples; use a display ROI to omit floor and wall."""
    camera = read_json(FRONT / "camera.json")
    K = np.asarray(camera["K"])
    T = np.asarray(camera["T_world_camera"])
    rgb = cv2.imdecode(np.frombuffer((FRONT/"rgb.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    depth = np.load(FRONT / "depth_m.npy")
    v, u = np.indices(depth.shape)
    camera_points = np.stack(((u-K[0, 2])*depth/K[0, 0], (v-K[1, 2])*depth/K[1, 1], depth), axis=-1)
    world = camera_points @ T[:3, :3].T + T[:3, 3]
    mask = ((world[:, :, 0] > 1.4) & (world[:, :, 0] < 2.65) &
            (np.abs(world[:, :, 1]) < .7) & (world[:, :, 2] > .015))
    mask[::2, ::2] = False
    xyz = world[mask][::3]
    colors = rgb[:, :, ::-1][mask][::3]
    # Fixed 9-byte records: three little-endian int16 mm values + three RGB bytes.
    packed = np.column_stack((np.rint(xyz*1000).astype("<i2").view("u1"), colors)).tobytes()
    return base64.b64encode(packed).decode("ascii"), len(xyz)


def img(path, alt):
    if not (ROOT/path).is_file():
        return ""
    return f'<img loading="lazy" src="../../{path}" alt="{html.escape(alt,quote=True)}">'


def active_views():
    cards = []
    records = []
    for filename, offset, title in [("initial",0.0,"首帧"),("revisit",-.50,"相机向右平移 0.50 m")]:
        rgb,depth,K,T = render_scene([Pallet()],view_y=offset,occlusion="partial")
        candidates = detect(rgb,depth,K,T)
        bgr=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
        records.append({"camera_y_m":offset,"candidates":candidates})
        for candidate in candidates:
            accepted = candidate["status"]=="ACCEPT"
            color = (80,163,49) if accepted else (42,143,229)
            x0,y0,x1,y1=candidate["bbox_xyxy"]
            cv2.rectangle(bgr,(x0,y0),(x1-1,y1-1),color,2)
            cv2.putText(bgr,candidate["status"],(x0,max(16,y0-8)),0,.52,(21,38,38),2,cv2.LINE_AA)
            if accepted:
                pts=(np.asarray(candidate["holes_world_m"])-T[:3,3])@T[:3,:3]
                uv=pts@K.T
                for point in np.rint(uv[:,:2]/uv[:,2:3]).astype(int):
                    cv2.drawMarker(bgr,tuple(point),(15,40,220),markerSize=12,thickness=2)
        ok, encoded = cv2.imencode(".png",bgr)
        if not ok: raise RuntimeError("PNG encoding failed")
        encoded.tofile(OUT/(filename+".png"))
        count=max((c["diagnostics"].get("verified_openings",0) for c in candidates),default=0)
        status="接受本帧位姿" if any(c["status"]=="ACCEPT" for c in candidates) else "请求重观测"
        cards.append(f'<figure><img src="{filename}.png" alt="{title}"><figcaption>{title} · 验证 {count} 个叉孔，{status}。</figcaption></figure>')
    (OUT/"reobservation.json").write_text(json.dumps(records,indent=2),encoding="utf-8")
    return '<div class="pair">'+''.join(cards)+'</div>'


def mission_frames():
    name,pallets,vis,instruction,expected = next(t for t in tasks() if t[0]=="blue-left")
    record=read_json(ROOT/"results"/"missions"/name/"record.json")
    rgb,z,K,T=render_scene(pallets,0.,vis)
    decision,selected,_=decide(rgb,z,K,T,Grounder.load(),instruction)
    if decision!=record["observations"][0]["decision"] or record["outcome"]!="DOCKED":
        raise ValueError("Replayed decision does not match mission record")
    sensor = cv2.imdecode(np.frombuffer((ROOT/"results/missions/blue-left/view-0.png").read_bytes(),np.uint8),1)
    # Orthographic xy projection; z slicing hides cargo above fork height.
    def project(xy):
        return np.rint(np.column_stack((542+165*(xy[:,0]-1.4),198-127*xy[:,1]))).astype("int32")
    canvas=np.full((420,960,3),(51,47,34),np.uint8)
    canvas[83:343,0:462]=cv2.resize(sensor,(462,260))
    cv2.putText(canvas,"CAMERA / selected "+decision,(16,51),0,.80,(226,241,231),1,cv2.LINE_AA)
    cv2.putText(canvas,"TOP VIEW / fork height",(497,51),0,.80,(226,241,231),1,cv2.LINE_AA)
    for origin,R,lo,hi,_ in solid_boxes(pallets,vis):
        if not lo[2]+origin[2]<selected["center_world_m"][2]<hi[2]+origin[2]:
            continue
        corners=np.array([[lo[0],lo[1],0.],[hi[0],lo[1],0.],[hi[0],hi[1],0.],[lo[0],hi[1],0.]]) @ R.T + origin
        cv2.fillConvexPoly(canvas,project(corners[:,:2]),(99,144,185))
    for frac in np.linspace(0.,1.,40):
        img=canvas.copy()
        tip=-START_GAP_M+frac*(.34+START_GAP_M)
        for corners in tine_corners(selected,tip):
            cv2.fillConvexPoly(img,project(corners),(110,201,145))
        state=("DOCKED" if frac==1. else "INSERTING" if tip>=0 else "APPROACHING")
        cv2.putText(img,state,(510,388),0,.75,(127,230,174),1,cv2.LINE_AA)
        yield img


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    packed, count = rgbd_cloud()
    data = read_json(ROOT / "sample_run/evaluation.json")
    entry = data["views"][0]
    benchmark = read_json(ROOT / "results/benchmark/summary.json")["single_pallet"]
    single,active = benchmark["single_view"], benchmark["one_revisit"]
    semantic = read_json(ROOT / "results/semantic/metrics.json")
    missions = read_json(ROOT/"results/missions/summary.json")
    mission=missions["blue-left"]
    achieved=mission["geometry_verdict"]["achieved_depth_m"]
    selection=mission["observations"][0]["decision"]
    still=next(frame for i,frame in enumerate(mission_frames()) if i==39)
    (OUT/"mission-docked.png").write_bytes(cv2.imencode(".png",still)[1].tobytes())
    content = (Path(__file__).with_name("viewer.html.in")).read_text(encoding="utf-8")
    replacements = {
        "CLOUD_DATA":packed, "POINT_COUNT":f"{count:,}",
        "RAW_Z":f"{entry['raw_hole_depth_m'][0]:.2f}",
        "ENTRY_Z":f"{entry['estimated_entry_depth_m'][0]:.3f}",
        "DELTA":f"{data['cross_view_difference_mm']:.2f}",
        "DOCKING":img("sample_run/docking-demo.gif","虚拟货叉实际碰撞回放"),
        "SEMANTIC":img("results/semantic/selection.png","候选托盘及目标选择结果"),
        "HOLES_JS":json.dumps(read_json(FRONT/"estimate.json")["centers_world"]),
        "DETECT":active_views(),
        "SEM_LINK":(f'<p class="tech">在 {semantic["test_scenes"]} 个留出场景的 {semantic["test_tasks"]} 条限定语法指令上，'
                    f'小型训练模型与规则分别答对 {semantic["grounder_correct"]} 和 {semantic["rule_correct"]} 条。'
                    '这组实验覆盖编号选择、目标缺失和指令歧义。<a href="../semantic/metrics.json">查看测量记录</a></p>'),
        "BENCH":(f'<p>同一批 {single["tasks"]} 个单托盘任务中，固定视角成功 {single["success"]} 次。允许一次重观测后成功 {active["success"]} 次，'
                 f'其中 {active["false_accepts"]} 次接受后对接失败。另有 {active["refused"]} 次仍拒绝执行。</p>'
                 '<p><a href="../benchmark/replays/recovered/record.json">换视角后恢复的样例</a>　'
                 '<a href="../benchmark/replays/refused/record.json">重观测后仍拒绝的样例</a>　'
                 '<a href="../benchmark/trials.csv">逐条场景记录</a></p>'),
        "MISSION":(f'<figure><img src="mission-docked.png" alt="蓝箱下的托盘与俯视货叉位置"><figcaption>任务指令「{html.escape(mission["instruction"])}」。'
                   f'选中 {selection}，双叉实测插入深度 {achieved:.3f} m，通过验收。</figcaption></figure>'
                   '<p><a href="../missions/blue-left/record.json">查看对接记录</a>　'
                   '<a href="../missions/reobserve/record.json">先重观测，再选目标</a>　'
                   '<a href="../missions/occluded/record.json">遮挡未解除，拒绝下发位姿</a></p>'),
    }
    for token, value in replacements.items():
        content = content.replace("{{"+token+"}}", value)
    (OUT/"index.html").write_text(content, encoding="utf-8")
    print("results/showcase/index.html")


if __name__ == "__main__":
    main()
