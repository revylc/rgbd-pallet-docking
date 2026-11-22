"""Export the project walkthrough as an MP4. Run after all benchmark scripts."""
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from showcase import ROOT, OUT, read_json, mission_frames


def read_rgb(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),1)


def fit(image, width, height):
    scale = min(width/image.shape[1],height/image.shape[0])
    return cv2.resize(image,(round(image.shape[1]*scale),round(image.shape[0]*scale)),
                      interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)


def centered(canvas, image, rect):
    x,y,w,h=rect
    tile = fit(image,w,h)
    dx,dy=(w-tile.shape[1])//2,(h-tile.shape[0])//2
    canvas[y+dy:y+dy+tile.shape[0],x+dx:x+dx+tile.shape[1]]=tile


def image_for(path, frame):
    if path.suffix != ".gif":
        return read_rgb(path)
    cap=cv2.VideoCapture(str(path))
    fps=cap.get(cv2.CAP_PROP_FPS) or 10
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if count<=0:
        cap.release()
        raise RuntimeError(f"Cannot decode {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(frame*fps/12)%count)
    ok,img=cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame at {frame} from {path}")
    return img


def main():
    bench=read_json(ROOT/"results/benchmark/summary.json")["single_pallet"]
    sem=read_json(ROOT/"results/semantic/metrics.json")
    auto,one=bench["single_view"],bench["one_revisit"]
    missions=read_json(ROOT/"results/missions/summary.json")
    result=missions["blue-left"]
    achieved=result["geometry_verdict"]["achieved_depth_m"]
    clip=list(mission_frames())
    shots=[
        ("01  /  RGB-D GEOMETRY","仿真相机测到的表面生成点云。青色圈标出了双叉孔的入口。", ["results/showcase/rgbd-cloud.gif"],5),
        ("02  /  DEPTH AMBIGUITY","中心像素测到 4.50 米后的墙。射线与木板前平面求交，恢复约 2 米处的入口。", ["sample_run/front/overlay.png"],5),
        ("03  /  ACTIVE OBSERVATION","同一个托盘，首帧只验证到一个孔。相机向右平移后，两个孔都能通过几何检查。", ["results/showcase/initial.png","results/showcase/revisit.png"],5),
        ("04  /  SWEPT-VOLUME TEST","虚拟货叉执行连续扫掠。对照带偏航误差的轨迹，可以看到首次接触位置。", ["sample_run/docking-demo.gif"],6),
        ("05  /  LANGUAGE GROUNDING",f'两个托盘与一条指令。受限语法下，训练模型和规则在 {sem["test_tasks"]} 项选择任务上均答对 {sem["grounder_correct"]} 项。', ["results/semantic/selection.png"],5),
        ("06  /  BENCHMARK",f'同一批 {one["tasks"]} 个任务中，单视角成功 {auto["success"]} 次，允许一次重观测后成功 {one["success"]} 次；其余 {one["refused"]} 次拒绝。', [],5),
        ("07  /  FULL MISSION",f'指令：{result["instruction"]}。模型选中 P1，双叉插入 {achieved:.3f} 米，超过 0.30 米验收线。', ["mission-replay"],7),
        ("08  /  REFUSAL",f'两次观测后仍被遮挡，任务状态为 {missions["occluded"]["outcome"]}，控制端不执行插入。', ["results/missions/occluded/view-0.png","results/missions/occluded/view-1.png"],4),
    ]
    def stamp(seconds):
        return f"00:{seconds//60:02d}:{seconds%60:02d},000"
    with tempfile.TemporaryDirectory() as dirname:
        directory=Path(dirname)
        seconds=0
        subtitles=[]
        for index,(_,subtitle,_,duration) in enumerate(shots,1):
            subtitles.append(f"{index}\n{stamp(seconds)} --> {stamp(seconds+duration)}\n{subtitle}\n")
            seconds+=duration
        (directory/"captions.srt").write_text("\n".join(subtitles),encoding="utf-8-sig")
        command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-pix_fmt","bgr24",
                 "-s","960x540","-framerate","12","-i","-","-vf",
                 "subtitles=captions.srt:force_style='FontName=Microsoft YaHei,FontSize=13,Outline=1.2,MarginV=19'",
                 "-vcodec","libx264","-preset","fast","-crf","19","-pix_fmt","yuv420p","-movflags","+faststart",str(OUT/"project-tour.mp4")]
        with subprocess.Popen(command,cwd=directory,stdin=subprocess.PIPE) as process:
            for title,subtitle,paths,duration in shots:
                for frame in range(duration*12):
                    canvas=np.full((540,960,3),(43,41,29),np.uint8)
                    cv2.putText(canvas,title,(26,41),0,.85,(233,242,230),2,cv2.LINE_AA)
                    cv2.line(canvas,(26,57),(934,57),(82,101,91))
                    for index,path in enumerate(paths):
                        section=920//len(paths)
                        tile=clip[min(len(clip)-1,frame)] if path=="mission-replay" else image_for(ROOT/path,frame)
                        centered(canvas,tile,(20+index*section,68,section-4,353))
                    if not paths:
                        for y,kind,group in [(160,"SINGLE VIEW",auto),(285,"ONE REVISIT",one)]:
                            cv2.putText(canvas,kind,(65,y-28),0,.85,(218,236,225),1,cv2.LINE_AA)
                            cv2.rectangle(canvas,(65,y),(875,y+32),(70,81,69),-1)
                            cv2.rectangle(canvas,(65,y),(65+round(810*group["success"]/group["tasks"]),y+32),(135,190,113),-1)
                            cv2.putText(canvas,f'{group["success"]} / {group["tasks"]}',(660,y-26),0,.84,(220,238,225),2,cv2.LINE_AA)
                    process.stdin.write(canvas.tobytes())
            process.stdin.close()
            if process.wait():
                raise RuntimeError("FFmpeg encoding failed")


if __name__ == "__main__":
    main()
