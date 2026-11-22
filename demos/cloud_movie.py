"""Make a short orbit around the recorded point cloud, using local FFmpeg."""
import base64
import math
import shutil
import subprocess

import cv2
import numpy as np

from showcase import FRONT, OUT, read_json, rgbd_cloud


def text(image, label, where, size=.5, color=(235,235,227)):
    cv2.putText(image, label, where, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def make_frames():
    packed, _ = rgbd_cloud()
    dtype = np.dtype([("xyz","<i2",(3,)),("rgb","u1",(3,))])
    points = np.frombuffer(base64.b64decode(packed), dtype)
    xyz = points["xyz"]/1000 - [2.1,0,.20]
    bgr = points["rgb"][:,::-1]
    holes = np.asarray(read_json(FRONT/"estimate.json")["centers_world"]) - [2.1,0,.2]
    source = cv2.imdecode(np.frombuffer((FRONT/"rgb.png").read_bytes(),np.uint8),1)
    width,height=822,340
    left = cv2.resize(source[95:290,120:510],(354,177),interpolation=cv2.INTER_AREA)
    for step in range(56):
        phase=2*math.pi*step/56
        yaw=.64*math.sin(phase)
        pitch=.19+.14*(1-math.cos(phase))
        cosa,sina=math.cos(yaw),math.sin(yaw)
        cosp,sinp=math.cos(pitch),math.sin(pitch)
        def project(vertices):
            X,Y,Z=vertices.T
            f=X*cosa+Y*sina
            return np.column_stack((610+317*(X*sina-Y*cosa),180-317*(Z*cosp+f*sinp),f*cosp-Z*sinp))
        canvas = np.full((height,width,3), (44,39,27),np.uint8)
        canvas[74:251,20:374]=left
        text(canvas,"RECORDED RGB", (20,38),.60)
        text(canvas,"METRIC SURFACE CLOUD", (402,38),.60)
        text(canvas,"fixed camera, single frame", (20,281),.48)
        text(canvas,"cyan rings: estimated entries", (402,281),.48)
        cv2.line(canvas,(20,296),(800,296),(78,93,79))
        text(canvas,"RGB-D PALLET DOCKING   /   SIMULATED OBSERVATION", (20,322),.46,(172,202,191))
        pos=project(xyz)
        for index in np.argsort(-pos[:,2]):
            x,y=np.rint(pos[index,:2]).astype(int)
            canvas[y:y+3,x:x+3]=bgr[index]
        for x,y,_ in project(holes):
            cv2.circle(canvas,(round(x),round(y)),5,(213,248,118),2,cv2.LINE_AA)
        yield canvas


def main():
    if not shutil.which("ffmpeg"):
        raise SystemExit("Install FFmpeg for GIF export. The interactive HTML needs no FFmpeg.")
    OUT.mkdir(parents=True,exist_ok=True)
    command=["ffmpeg","-hide_banner","-loglevel","error","-y","-f","rawvideo","-vcodec","rawvideo",
             "-pix_fmt","bgr24","-s","822x340","-framerate","12","-i","-","-filter_complex",
             "[0:v]split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=none",
             "-loop","0",str(OUT/"rgbd-cloud.gif")]
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for frame in make_frames():
            process.stdin.write(frame.tobytes())
        process.stdin.close()
        if process.wait():
            raise SystemExit("FFmpeg encoding failed")


if __name__ == "__main__":
    main()
