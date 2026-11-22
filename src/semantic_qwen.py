"""Optional adapter for local Qwen2-VL weights. The default demo uses Grounder."""
from time import perf_counter

import cv2


class LocalQwenSelector:
    def __init__(self, model_directory):
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_directory, local_files_only=True, torch_dtype=dtype).to(device).eval()

    def select(self, rgb, candidates, instruction):
        from PIL import Image

        start = perf_counter()
        marked = rgb.copy()
        allowed = {c["id"] for c in candidates} | {"ABSENT","AMBIGUOUS"}
        for candidate in candidates:
            x,y,x1,y1 = candidate["bbox_xyxy"]
            cv2.rectangle(marked,(x,y),(x1,y1),(255,255,255),2)
            cv2.putText(marked,candidate["id"],(x+3,y+18),0,.6,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(marked,candidate["id"],(x+3,y+18),0,.6,(255,255,255),1,cv2.LINE_AA)
        prompt = ("图中托盘标有编号。按指令选择托盘。只输出一个候选编号；没有目标回答 ABSENT；"
                  "多个目标同时符合回答 AMBIGUOUS。可选答案："+"，".join(sorted(allowed))
                  +"。指令："+instruction)
        messages = [{"role":"user","content":[{"type":"image"},{"type":"text","text":prompt}]}]
        text = self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs = self.processor(text=[text],images=[Image.fromarray(marked)],return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs,max_new_tokens=16,do_sample=False)
        answer = self.processor.batch_decode(output[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)[0].strip()
        return {"decision":answer if answer in allowed else "UNCERTAIN", "raw":answer,
                "latency_ms":(perf_counter()-start)*1000}
