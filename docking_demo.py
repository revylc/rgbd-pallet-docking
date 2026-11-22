"""Animate a successful command and an injected yaw error against one pallet."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.docking import START_GAP_M, tine_corners, validate


def _font(size):
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_demo(path):
    pose = {"center_world_m": [2., 0., .25], "yaw_deg": 12.}
    commands = [dict(pose), dict(pose, yaw_deg=32.)]
    results = [validate(pose, cmd) for cmd in commands]
    angle = np.deg2rad(pose["yaw_deg"])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    big, small, caption = _font(26), _font(16), _font(13)
    frames = []
    for timeline in np.linspace(0., 1., 37):
        picture = Image.new("RGB", (850, 380), "#142936")
        draw = ImageDraw.Draw(picture)
        draw.text((22, 14), "Virtual fork insertion", font=big, fill="white")
        draw.text((22, 52), "Top-down slice at fork height   /   Two solid tines   /   Continuous collision test",
                  font=small, fill="#c4d4da")
        for panel, (command, result) in enumerate(zip(commands, results)):
            left = 15 + panel*420
            draw.rounded_rectangle((left, 87, left+402, 362), radius=8, fill="#f1f3ef")
            title = "Aligned" if panel == 0 else "Injected yaw error  +20 deg"
            draw.text((left+12, 98), title, font=small, fill="#213b46")
            def project(xy):
                return [(left+219 + 170*x, 223-170*y) for x, y in xy]
            for low, high in ((-.50,-.41),(-.09,.09),(.41,.50)):
                polygon = np.array([[0,low],[.4,low],[.4,high],[0,high]]) @ rotation.T
                draw.polygon(project(polygon), fill="#bf956a", outline="#7d5e43")
            # The distant line marks the required insertion depth in the true frame.
            line = np.array([[.30,-.50],[.30,.50]]) @ rotation.T
            draw.line(project(line), fill="#788d91", width=2)
            stopped = min(timeline, result["travel_fraction"])
            tip = -START_GAP_M + stopped*(START_GAP_M+.34)
            collided = result["status"] == "collision" and timeline >= result["travel_fraction"]
            for corners in tine_corners(command, tip):
                draw.polygon(project(corners-np.array(pose["center_world_m"][:2])),
                             fill="#bc4838" if collided else "#25838b", outline="#1e5159")
            status = "APPROACHING" if tip < 0 else "INSERTING"
            if collided:
                status = "CONTACT  /  " + result["reason"]
            elif timeline >= 1.:
                status = "PASS  /  both tips passed 0.30 m"
            draw.text((left+12,328),status,font=caption,fill="#9b392b" if collided else "#175b61")
        frames.append(picture)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=[80]*(len(frames)-1)+[1800], optimize=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("sample_run/docking-demo.gif"))
    args = parser.parse_args()
    for row in make_demo(args.output):
        print(row["status"], row["reason"])
    print(args.output)
