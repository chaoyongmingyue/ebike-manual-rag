import urllib.request
import json
import base64
import time
from pathlib import Path

class VLMClient:
    """Ollama Qwen3-VL client for image description."""

    def __init__(self, model="qwen3-vl:4b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/api/chat"

    def describe(self, image_path: str, prompt: str = "") -> str:
        """Send an image to the VLM and get description."""
        default_prompt = (
            "请详细描述这张图片中的内容。如果包含文字，请完整提取所有文字信息。"
            "如果是表格，请输出完整的Markdown表格格式。"
            "如果是流程图、原理图或示意图，请描述其结构和含义。"
            "如果是产品照片或插图，请描述图中展示的内容。"
            "请用中文回答。"
        )
        user_prompt = prompt or default_prompt

        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                    "images": [img_b64],
                }
            ],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 2048},
            "keep_alive": "5m",
        }

        req = urllib.request.Request(
            self.api_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        max_retries = 2
        for attempt in range(max_retries):
            try:
                resp = urllib.request.urlopen(req, timeout=180)
                result = json.loads(resp.read().decode("utf-8"))
                return result["message"]["content"].strip()
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3)
                    continue
                return f"[VLM Error: {e}]"
