"""Check MiniMax workflow discovery; optionally run a small reference-to-video test."""

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from PIL import Image, ImageDraw


VERSION = "1.0.0"
REPO = Path(__file__).resolve().parents[1]


def request(base, endpoint, data=None):
    payload = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base + endpoint, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode()) from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8188")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--version", action="version", version=VERSION)
    args = parser.parse_args()
    workflow = json.loads((REPO / "video_minimax_h3_r2v.json").read_text(encoding="utf-8"))
    info = request(args.base_url, "/object_info")
    links = {link[0]: [str(link[1]), link[2]] for link in workflow["links"]}
    prompt = {}
    for node in workflow["nodes"]:
        if node["type"] == "MarkdownNote":
            continue
        schema = info[node["type"]]["input"]
        names = set(schema.get("required", {})) | set(schema.get("optional", {}))
        inputs = {k: v for k, v in node.get("widgets_values_named", {}).items() if k.split(".")[0] in names}
        for socket in node.get("inputs", []):
            if socket.get("link") is not None:
                inputs[socket["name"]] = links[socket["link"]]
        prompt[str(node["id"])] = {"class_type": node["type"], "inputs": inputs}
        for model in node.get("properties", {}).get("models", []):
            field = {"UNETLoader": "unet_name", "CLIPLoader": "clip_name", "VAELoader": "vae_name", "LoraLoaderModelOnly": "lora_name"}[node["type"]]
            if model["name"] not in schema["required"][field][0]:
                raise ValueError(f"Model unavailable in ComfyUI: {model['name']}")
            print(f"DISCOVERED {model['name']}", flush=True)
    if not args.generate:
        print("SUCCESS: all workflow node types and five model files are available.")
        return
    queue = request(args.base_url, "/queue")
    if queue["queue_running"] or queue["queue_pending"]:
        raise RuntimeError("ComfyUI has queued work; run this test when idle")
    fixture_name = "minimax_h3_smoke_reference.png"
    fixture = Image.new("RGB", (256, 256), "#d9e4ee")
    ImageDraw.Draw(fixture).ellipse((72, 72, 184, 184), fill="#c52d35")
    fixture.save(REPO / "input" / fixture_name)
    for node_id in ["137", "139"]:
        prompt[node_id]["inputs"]["image"] = fixture_name
    prompt["136"]["inputs"].update(width=256, height=256, length=22)
    prompt["138"]["inputs"]["value"] = "integrated_multimodal_description: [Shot 1] The red ball in <Picture 1> and <Picture 2> moves slowly to the right on a pale blue background. A fixed camera watches the ball. overall_soundscape: A soft rolling sound. non_diegetic_music: None."
    prompt["146"]["inputs"]["value"] = True
    prompt["129"]["inputs"]["noise_seed"] = 12345
    prompt["92"]["inputs"]["filename_prefix"] = "validation/minimax_h3_nvfp4_smoke"
    result = request(args.base_url, "/prompt", {"prompt": prompt, "client_id": "minimax-h3-verification"})
    prompt_id = result["prompt_id"]
    print(f"SUBMITTED {prompt_id}: 256x256, 22 frames, Turbo 4 steps", flush=True)
    while True:
        history = request(args.base_url, "/history/" + prompt_id)
        if prompt_id in history:
            record = history[prompt_id]
            (REPO / "temp" / "minimax_h3_smoke_result.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(json.dumps(record["status"]), flush=True)
            if record["status"]["status_str"] != "success":
                raise RuntimeError("Generation failed; see temp/minimax_h3_smoke_result.json")
            print("SUCCESS", json.dumps(record["outputs"]), flush=True)
            return
        print("RUNNING: waiting for the test generation", flush=True)
        time.sleep(20)


if __name__ == "__main__":
    main()
