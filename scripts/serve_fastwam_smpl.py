import argparse
import base64
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from hydra.utils import instantiate
from PIL import Image

from infer_ego import _build_cfg, _denormalize_action
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from fastwam.runtime import build_datasets


def _numpy_to_json(arr: np.ndarray) -> dict[str, Any]:
    arr = np.ascontiguousarray(arr)
    return {
        "__numpy__": base64.b64encode(arr.tobytes()).decode("ascii"),
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
    }


def _json_to_numpy(obj: Any) -> np.ndarray:
    if isinstance(obj, dict) and "__numpy__" in obj:
        raw = base64.b64decode(obj["__numpy__"])
        return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return np.asarray(obj)


def _decode_image(payload: Any) -> Image.Image:
    if isinstance(payload, dict) and "__numpy__" in payload:
        arr = _json_to_numpy(payload).astype(np.uint8, copy=False)
        return Image.fromarray(arr).convert("RGB")
    if isinstance(payload, str):
        raw = base64.b64decode(payload)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.asarray(payload, dtype=np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _image_to_input_tensor(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    image = image.resize((size[1], size[0]), Image.BILINEAR)
    arr = np.asarray(image, dtype=np.uint8)
    tensor = torch.from_numpy(arr).permute(2, 0, 1).to(dtype=torch.float32)
    return (tensor / 127.5 - 1.0).unsqueeze(0)


def _encode_image_jpeg(image: Image.Image, quality: int = 85) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=int(quality))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _load_cached_context(cache_dir: Path, context_len: int, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cache_path = cache_dir / f"{hashed}.t5_len{context_len}.wan22ti2v5b.pt"
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing text embedding cache: {cache_path}. "
            "Run scripts/precompute_text_embeds.py for this instruction first."
        )
    payload = torch.load(cache_path, map_location="cpu")
    return payload["context"], payload["mask"].bool()


class FastWAMSMPLServer:
    def __init__(
        self,
        task: str,
        checkpoint: Path,
        mode: str,
        device: str,
        mixed_precision: str,
        num_inference_steps: int,
        action_horizon: int | None,
        instruction: str,
        image_size: tuple[int, int] | None,
        rand_device: str,
        seed: int,
    ):
        self.task = task
        self.checkpoint = checkpoint
        self.mode = str(mode)
        self.device = device
        self.num_inference_steps = int(num_inference_steps)
        self.rand_device = rand_device
        self.seed = int(seed)

        cfg = _build_cfg(task)
        self.cfg = cfg
        model_dtype = {
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[mixed_precision]

        self.model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint))
        self.model = self.model.to(device).eval()

        train_ds, _val_ds = build_datasets(cfg.data)
        self.processor = train_ds.lerobot_dataset.processor
        self.action_dim = int(cfg.data.train.processor.action_output_dim)
        self.num_frames = int(cfg.data.train.num_frames)
        self.action_horizon = int(action_horizon if action_horizon is not None else cfg.data.train.num_frames - 1)
        if image_size is None:
            h, w = cfg.data.train.video_size
            self.image_size = (int(h), int(w))
        else:
            self.image_size = image_size

        self.instruction = instruction
        self.prompt = DEFAULT_PROMPT.format(task=instruction)
        cache_dir = Path(str(cfg.data.train.text_embedding_cache_dir)).expanduser()
        self.context, self.context_mask = _load_cached_context(
            cache_dir=cache_dir,
            context_len=int(cfg.data.train.context_len),
            prompt=self.prompt,
        )

        self._print_cuda_memory("after_load")

    def _print_cuda_memory(self, label: str) -> None:
        if not torch.cuda.is_available() or not str(self.device).startswith("cuda"):
            return
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        peak = torch.cuda.max_memory_allocated() / 1024**3
        print(f"[fastwam_smpl] cuda_memory {label}: allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB peak={peak:.2f}GiB")

    @torch.inference_mode()
    def predict(
        self,
        image: Image.Image,
        instruction: str | None = None,
        return_video: bool = False,
        seed: int | None = None,
    ) -> tuple[np.ndarray, dict[str, Any], list[str] | None]:
        if instruction is not None and instruction != self.instruction:
            prompt = DEFAULT_PROMPT.format(task=instruction)
            cache_dir = Path(str(self.cfg.data.train.text_embedding_cache_dir)).expanduser()
            context, context_mask = _load_cached_context(
                cache_dir=cache_dir,
                context_len=int(self.cfg.data.train.context_len),
                prompt=prompt,
            )
        else:
            context, context_mask = self.context, self.context_mask

        input_image = _image_to_input_tensor(image, self.image_size)
        t0 = time.perf_counter()
        infer_seed = self.seed if seed is None else int(seed)
        if self.mode == "action_only":
            out = self.model.infer_action(
                prompt=None,
                input_image=input_image.to(device=self.model.device, dtype=self.model.torch_dtype),
                action_horizon=self.action_horizon,
                context=context.to(device=self.model.device, dtype=self.model.torch_dtype),
                context_mask=context_mask.to(device=self.model.device, dtype=torch.bool),
                num_inference_steps=self.num_inference_steps,
                seed=infer_seed,
                rand_device=self.rand_device,
                tiled=False,
            )
            pred_video = None
        elif self.mode == "video_action":
            out = self.model.infer_joint(
                prompt=None,
                input_image=input_image.to(device=self.model.device, dtype=self.model.torch_dtype),
                num_video_frames=self.num_frames,
                action_horizon=self.action_horizon,
                action=None,
                context=context.to(device=self.model.device, dtype=self.model.torch_dtype),
                context_mask=context_mask.to(device=self.model.device, dtype=torch.bool),
                negative_prompt="",
                text_cfg_scale=1.0,
                num_inference_steps=self.num_inference_steps,
                seed=infer_seed,
                rand_device=self.rand_device,
                tiled=False,
                test_action_with_infer_action=False,
            )
            pred_video = out.get("video")
        else:
            raise ValueError(f"Unsupported server mode: {self.mode}")
        pred_action = out["action"].detach().cpu()
        denorm = _denormalize_action(self.processor, pred_action, proprio_txd=None)
        action = denorm.detach().cpu().numpy().astype(np.float32, copy=False)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        video_payload = None
        if return_video and pred_video is not None:
            video_payload = [_encode_image_jpeg(frame) for frame in pred_video]
        meta = {
            "mode": self.mode,
            "latency_ms": latency_ms,
            "action_horizon": int(action.shape[0]),
            "action_dim": int(action.shape[1]),
            "num_frames": self.num_frames,
            "num_inference_steps": self.num_inference_steps,
            "seed": int(infer_seed),
            "predicted_video": pred_video is not None,
            "returned_video": video_payload is not None,
        }
        return action, meta, video_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="FastWAM online SMPL/action inference HTTP server.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--mode",
        choices=["action_only", "video_action"],
        default="action_only",
        help="action_only calls infer_action and does not predict video; video_action calls infer_joint and predicts video+action.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--instruction", default="Pick up the yellow clothes on the bed and place them into the white basket on the right.")
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--rand-device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--warmup", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse
    except ImportError as exc:
        raise ImportError("This server requires fastapi and uvicorn.") from exc

    image_size = None
    if args.height is not None or args.width is not None:
        if args.height is None or args.width is None:
            raise ValueError("--height and --width must be provided together.")
        image_size = (int(args.height), int(args.width))

    server = FastWAMSMPLServer(
        task=args.task,
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        mode=args.mode,
        device=args.device,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_inference_steps,
        action_horizon=args.action_horizon,
        instruction=args.instruction,
        image_size=image_size,
        rand_device=args.rand_device,
        seed=args.seed,
    )

    if args.warmup:
        dummy = Image.fromarray(np.zeros((server.image_size[0], server.image_size[1], 3), dtype=np.uint8))
        action, meta, _video = server.predict(dummy)
        print(f"[fastwam_smpl] warmup action_shape={action.shape} meta={meta}")
        server._print_cuda_memory("after_warmup")

    app = FastAPI(title="FastWAM SMPL Server")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "task": server.task,
            "mode": server.mode,
            "action_dim": server.action_dim,
            "action_horizon": server.action_horizon,
            "num_frames": server.num_frames,
        }

    @app.post("/act")
    async def act(payload: dict[str, Any]) -> JSONResponse:
        try:
            image_payload = payload.get("image", {})
            if isinstance(image_payload, dict) and "egocentric" in image_payload:
                image = _decode_image(image_payload["egocentric"])
            else:
                image = _decode_image(image_payload)
            instruction = payload.get("instruction")
            action, meta, video = server.predict(
                image=image,
                instruction=instruction,
                return_video=bool(payload.get("return_video", False)),
                seed=payload.get("seed"),
            )
            response = {"action": _numpy_to_json(action), "err": 0.0, "meta": meta}
            if video is not None:
                response["video_jpeg"] = video
            return JSONResponse(content=response)
        except Exception as exc:
            return JSONResponse(content={"error": str(exc)}, status_code=400)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
