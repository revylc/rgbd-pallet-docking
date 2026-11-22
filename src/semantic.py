"""A small task-trained RGB/text grounding network and a separate colour-rule baseline."""
import re
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np


DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "models" / "grounder_v1.npz"
WORDS = ("蓝", "红", "左", "右")
# Known task grammar is intentionally narrow. Reject unrecognised relations instead of
# silently discarding a negation or pretending to understand an open-ended command.
PATTERN = re.compile(r"^(?:请|帮我|请帮我)?(?:取|选|选择|找|定位|拿)?(?:出|到|一下)?"
                     r"(?:画面|图中|图像中|画面中)?(?:最)?(?:左|右)?(?:侧|边|方)?(?:的)?"
                     r"(?:蓝|红)?(?:色)?(?:料箱|箱子|箱)?(?:下方|下面|下|底下)?(?:的|那个)?"
                     r"(?:栈板|托盘)[。！!？?]?$")


def encode_text(instruction):
    text = "".join(instruction.split())
    if not PATTERN.fullmatch(text):
        return None
    return np.array([float(w in text) for w in WORDS], dtype=np.float32)


def cargo_patch(rgb, bbox):
    x0, y0, x1, y1 = bbox
    width, height = x1 - x0, y1 - y0
    # Sample the region above the top deck where the load is visible. Include some
    # margin because the front face of an elevated load shifts under perspective.
    x0, x1 = int(x0 + width * .13), int(x1 - width * .13)
    y0, y1 = int(y0 - height * .83), int(y0 + height * .05)
    return rgb[max(0, y0):min(rgb.shape[0], y1), max(0, x0):min(rgb.shape[1], x1)]


def ordered_candidates(candidates):
    return sorted(candidates, key=lambda c: c["bbox_xyxy"][0] + c["bbox_xyxy"][2])


def image_features(rgb, candidates):
    ordered = ordered_candidates(candidates)
    features = []
    for i, c in enumerate(ordered):
        patch = cargo_patch(rgb, c["bbox_xyxy"])
        if patch.size == 0:
            return ordered, None
        # Preserve coarse spatial appearance with a 4x4 RGB grid. All vision
        # projections downstream of this pooling operation are learned from pairs.
        grid = cv2.resize(patch, (4, 4), interpolation=cv2.INTER_AREA).reshape(-1)/127.5 - 1.
        side = (2*i/(len(ordered)-1) - 1.) if len(ordered) > 1 else 0.
        features.append(np.append(grid, side))
    return ordered, np.asarray(features, dtype=np.float32)


def mlp(params, images, text):
    visual = np.tanh(images @ params["Vi"] + params["bi"])
    query = np.tanh(text @ params["Wt"] + params["bt"])
    fusion = np.concatenate((visual, query, visual * query), axis=1)
    hidden = np.tanh(fusion @ params["Wh"] + params["bh"])
    logits = (hidden @ params["Wo"] + params["bo"]).ravel()
    return 1/(1+np.exp(-np.clip(logits, -30, 30))), (visual, query, fusion, hidden)


def decision_from_scores(scores):
    if any(.25 < s < .75 for s in scores.values()):
        return "UNCERTAIN"
    found = [key for key, value in scores.items() if value >= .75]
    if not found:
        return "ABSENT"
    return found[0] if len(found) == 1 else "AMBIGUOUS"


def result(decision, scores, start):
    return {"decision":decision, "scores":scores, "latency_ms":(perf_counter()-start)*1000}


class Grounder:
    def __init__(self, params):
        self.params = params

    @classmethod
    def load(cls, path=DEFAULT_WEIGHTS):
        with np.load(path, allow_pickle=False) as archive:
            return cls({k:archive[k] for k in archive.files})

    def select(self, rgb, candidates, instruction):
        start = perf_counter()
        text = encode_text(instruction)
        if text is None:
            return result("UNSUPPORTED", {}, start)
        if not candidates:
            return result("ABSENT", {}, start)
        ordered, images = image_features(rgb, candidates)
        if images is None:
            return result("UNCERTAIN", {}, start)
        scores, _ = mlp(self.params, images, np.tile(text, (len(images), 1)))
        by_id = {c["id"]:float(s) for c,s in zip(ordered,scores)}
        return result(decision_from_scores(by_id), by_id, start)


def select_by_rule(rgb, candidates, instruction):
    start = perf_counter()
    text = encode_text(instruction)
    if text is None:
        return result("UNSUPPORTED", {}, start)
    blue, red, left, right = text
    scores = {}
    ordered = ordered_candidates(candidates)
    for i, c in enumerate(ordered):
        patch = cargo_patch(rgb,c["bbox_xyxy"])
        if not patch.size:
            return result("UNCERTAIN", {}, start)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        hue = hsv[:,:,0]
        saturated = hsv[:,:,1] > 90
        blue_fraction = np.mean(saturated & (hue > 90) & (hue < 135))
        red_fraction = np.mean(saturated & ((hue < 9) | (hue > 170)))
        selected = ((not blue or blue_fraction > .12) and (not red or red_fraction > .12)
                    and (not left or i == 0) and (not right or i == len(ordered)-1))
        scores[c["id"]] = float(selected)
    return result(decision_from_scores(scores), scores, start)
