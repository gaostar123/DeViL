import sys
import os
import re
import argparse

sys.path.append("./")

from devil import disable_torch_init, model_init, mm_infer
from devil.mm_utils import load_video_new, load_images
from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
import numpy as np
import cv2

def _normalize_spatial_shapes(spatial_shapes):
    if spatial_shapes is None:
        return []
    if isinstance(spatial_shapes, torch.Tensor):
        if spatial_shapes.ndim == 2 and spatial_shapes.size(1) >= 2:
            return [(int(h), int(w)) for h, w in spatial_shapes[:, :2].tolist()]
        return []
    if isinstance(spatial_shapes, (list, tuple)):
        out = []
        for s in spatial_shapes:
            if isinstance(s, torch.Tensor):
                h = int(s[0].item())
                w = int(s[1].item())
            else:
                h = int(s[0])
                w = int(s[1])
            out.append((h, w))
        return out
    return []


def build_attn_heatmap(image_pil, attn_maps, spatial_shapes, level_idx=2):

    if attn_maps is None:
        return None, None

    attn_maps = [
        am for am in attn_maps
        if isinstance(am, torch.Tensor) and am.ndim == 2 and am.size(0) == 1
    ]
    if len(attn_maps) == 0:
        return None, None

    spatial_shapes = _normalize_spatial_shapes(spatial_shapes)
    if len(spatial_shapes) == 0:
        return None, None
    if level_idx < 0 or level_idx >= len(spatial_shapes):
        return None, None

    try:
        avg_attn = torch.stack(attn_maps, dim=0).mean(dim=0)
        vec = avg_attn.squeeze(0)

        offset = 0
        h_feat, w_feat = None, None
        for li, (h, w) in enumerate(spatial_shapes):
            n = h * w
            if li == level_idx:
                h_feat, w_feat = h, w
                attn_l = vec[offset:offset + n].reshape(h_feat, w_feat)
                break
            offset += n
        if h_feat is None:
            return None, None

        W_img, H_img = image_pil.size  # PIL: (W, H)
        a_min, a_max = float(attn_l.min()), float(attn_l.max())
        attn_l = (attn_l - a_min) / (a_max - a_min + 1e-6)

        attn_up = F.interpolate(
            attn_l.unsqueeze(0).unsqueeze(0),
            size=(H_img, W_img),
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)

        heatmap_gray = (attn_up.detach().cpu().numpy() * 255.0).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        heatmap_img = Image.fromarray(heatmap_color).convert("RGB")
        blended = Image.blend(image_pil.convert("RGB"), heatmap_img, alpha=0.5)
        return heatmap_img, blended
    except Exception:
        return None, None


def parse_time_range(output_text):

    s = output_text if isinstance(output_text, str) else str(output_text)
    s = s.replace("–", "-")

    patterns = [
        r'from\s+(\d+(?:\.\d+)?)\s+to\s+(\d+(?:\.\d+)?)',
        r'between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)',
        r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)'
    ]
    for pat in patterns:
        m = re.search(pat, s, flags=re.IGNORECASE)
        if m:
            s_val, e_val = float(m.group(1)), float(m.group(2))
            if e_val < s_val:
                s_val, e_val = e_val, s_val
            return s_val, e_val
    return None


def save_text_output(output, output_dir, media_path, query):
    out_path = os.path.join(output_dir, "output.txt")
    with open(out_path, "w") as f:
        f.write(str(output) + "\n")
        if media_path:
            f.write(str(media_path) + "\n")
        if query:
            f.write(str(query) + "\n")
    print(f"[INFO] Text output saved to: {out_path}")


def run_image(model, processor, image_path, query, output_dir,
              visualize_attention=False, top_k=1):
    modal = "image"
    frames = load_images(image_path)[0]

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        }
    ]

    inputs = processor(
        images=[frames],
        text=conversation,
        merge_size=1,
        return_tensors="pt",
    )
    inputs["images"] = [[frames]]

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal
    )
    print("Model Output:", output)
    save_text_output(output, output_dir, image_path, query)

    has_dino = hasattr(model, "infer_dino_output") and model.infer_dino_output is not None
    boxes_saved = False

    if has_dino:
        dino_output = model.infer_dino_output
        if isinstance(dino_output, dict) and "pred_logits" in dino_output and "pred_boxes" in dino_output:
            try:
                pred_logits = dino_output["pred_logits"][0]  # [T, Q, C]
                pred_boxes = dino_output["pred_boxes"][0]    # [T, Q, 4]

                if (pred_logits.ndim == 3 and pred_boxes.ndim == 3
                        and pred_logits.shape[0] > 0
                        and pred_logits.shape[:2] == pred_boxes.shape[:2]
                        and pred_logits.shape[1] > 0):
                    pred_scores = pred_logits.sigmoid()      # [T, Q, C]
                    pred_scores = pred_scores.mean(0)        # [Q, C]
                    max_scores, _ = pred_scores.max(-1)      # [Q]

                    k = min(top_k, max_scores.size(0))
                    if k > 0:
                        _, top_indices = max_scores.topk(k)

                        img = Image.open(image_path).convert("RGB")
                        draw = ImageDraw.Draw(img)
                        w, h = img.size
                        colors = ["red", "blue", "green", "yellow", "purple"]

                        for idx, box_idx in enumerate(top_indices):
                            pred_box = pred_boxes[0][box_idx].cpu().numpy()
                            x_center, y_center, width, height = pred_box
                            x1 = int((x_center - width / 2) * w)
                            y1 = int((y_center - height / 2) * h)
                            x2 = int((x_center + width / 2) * w)
                            y2 = int((y_center + height / 2) * h)
                            draw.rectangle((x1, y1, x2, y2),
                                           outline=colors[idx % len(colors)],
                                           width=5)

                        base_name = os.path.splitext(os.path.basename(image_path))[0]
                        output_path = os.path.join(output_dir, f"{base_name}_boxed.jpg")
                        img.save(output_path)
                        boxes_saved = True
                        print(f"[INFO] Image with top {k} bounding boxes saved to: {output_path}")
                else:
                    print("[WARN] DINO outputs exist but shapes are invalid. Skip box visualization.")
            except Exception as e:
                print(f"[WARN] Exception when processing DINO boxes: {e}. Only text output will be saved.")
        else:
            print("[WARN] infer_dino_output has no 'pred_logits' or 'pred_boxes'. Skip box visualization.")
    else:
        print("[INFO] No DINO detector outputs found. Only text output is saved.")

    if visualize_attention:
        can_vis_attn = (
            hasattr(model.get_model(), "g_dino") and
            hasattr(model.get_model().g_dino, "attention_maps") and
            isinstance(model.get_model().g_dino.attention_maps, list) and
            len(model.get_model().g_dino.attention_maps) > 0 and
            hasattr(model.get_model().g_dino, "feature_spatial_shapes") and
            model.get_model().g_dino.feature_spatial_shapes is not None
        )

        if not can_vis_attn:
            print("[INFO] Could not find attention maps to visualize for image.")
            return

        attn_cache_raw = model.get_model().g_dino.attention_maps
        spatial_shapes_any = model.get_model().g_dino.feature_spatial_shapes
        spatial_shapes = _normalize_spatial_shapes(spatial_shapes_any)
        if len(spatial_shapes) == 0 or not attn_cache_raw:
            print("[WARN] attention maps or spatial shapes invalid; skip attention visualization.")
            return

        per_layer_vecs = []
        for layer_attn in attn_cache_raw:
            if isinstance(layer_attn, torch.Tensor) and layer_attn.dim() == 2 and layer_attn.size(0) >= 1:
                per_layer_vecs.append(layer_attn[0:1])

        if len(per_layer_vecs) > 0:
            src_img = Image.open(image_path).convert("RGB")
            _, blended_img = build_attn_heatmap(src_img, per_layer_vecs, spatial_shapes, level_idx=2)
            if blended_img is not None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                attention_output_path = os.path.join(output_dir, f"{base_name}_attention_lvl2.jpg")
                blended_img.save(attention_output_path)
                print(f"[INFO] Image with attention map saved to: {attention_output_path}")
            else:
                print("[WARN] build_attn_heatmap returned None; skip attention visualization.")
        else:
            print("[WARN] No per-layer attention vectors; skip attention visualization.")



def run_video(model, processor, video_path, query, output_dir,
              visualize_attention=False):
    modal = "video"
    frames, timestamps, imags = load_video_new(video_path, make_pil=True)

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "timestamps": timestamps, "num_frames": len(frames)},
                {"type": "text", "text": query},
            ]
        }
    ]

    inputs = processor(
        images=[frames],
        text=conversation,
        merge_size=2,
        return_tensors="pt",
    )
    inputs["images"] = [imags]

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal
    )
    print("Model Output:", output)
    save_text_output(output, output_dir, video_path, query)

    do_attn = False
    attn_cache_raw, spatial_shapes = None, None
    if visualize_attention:
        do_attn = (
            hasattr(model.get_model(), "g_dino") and
            hasattr(model.get_model().g_dino, "attention_maps") and
            isinstance(model.get_model().g_dino.attention_maps, list) and
            len(model.get_model().g_dino.attention_maps) > 0 and
            hasattr(model.get_model().g_dino, "feature_spatial_shapes") and
            model.get_model().g_dino.feature_spatial_shapes is not None
        )
        if do_attn:
            attn_cache_raw = model.get_model().g_dino.attention_maps
            spatial_shapes_any = model.get_model().g_dino.feature_spatial_shapes
            spatial_shapes = _normalize_spatial_shapes(spatial_shapes_any)
            if len(spatial_shapes) == 0 or not attn_cache_raw:
                print("[WARN] attention maps or spatial shapes invalid; skip attention visualization.")
                do_attn = False
        else:
            print("[INFO] Could not find attention maps to visualize for video.")

    if not hasattr(model, "infer_dino_output") or model.infer_dino_output is None:
        print("[INFO] DINO detector outputs not found. Only text output is saved.")
        return

    dino_output = model.infer_dino_output
    if not (isinstance(dino_output, dict) and "pred_logits" in dino_output and "pred_boxes" in dino_output):
        print("[WARN] infer_dino_output has no 'pred_logits' or 'pred_boxes'. Only text output is saved.")
        return

    try:
        pred_logits = dino_output["pred_logits"][0]  # [T, Q, C]
        pred_boxes = dino_output["pred_boxes"][0]    # [T, Q, 4]
    except Exception as e:
        print(f"[WARN] Failed to read DINO outputs: {e}. Only text output is saved.")
        return

    if not (pred_logits.ndim == 3 and pred_boxes.ndim == 3
            and pred_logits.shape[:2] == pred_boxes.shape[:2]
            and pred_logits.shape[0] > 0 and pred_logits.shape[1] > 0):
        print("[WARN] DINO outputs shapes invalid. Only text output is saved.")
        return

    time_range = parse_time_range(output)
    relevant_frame_indices = []
    if time_range is not None:
        start_time, end_time = time_range
        print(f"[INFO] Parsed time range: Start={start_time:.2f}s, End={end_time:.2f}s. Processing relevant frames.")
        for i, ts in enumerate(timestamps):
            if start_time <= ts <= end_time:
                relevant_frame_indices.append(i)
    else:
        print("[WARN] Could not parse start and end timestamps from the model's output. Processing the ENTIRE video.")
        relevant_frame_indices = list(range(len(imags)))

    if not relevant_frame_indices:
        print("[WARN] No frames to process.")
        return

    num_frames_dino = pred_logits.shape[0]
    relevant_frame_indices = [idx for idx in relevant_frame_indices if idx < num_frames_dino]
    if not relevant_frame_indices:
        print("[WARN] No valid frames after aligning with DINO outputs.")
        return

    print(f"[INFO] Found {len(relevant_frame_indices)} frames to process.")
    print(f"[INFO] Frame indices to process: {relevant_frame_indices}")

    with torch.no_grad():
        all_scores = pred_logits.sigmoid()                     # [T, Q, C]
        max_scores_per_query_per_frame, _ = all_scores.max(-1) # [T, Q]
        tube_scores = max_scores_per_query_per_frame[relevant_frame_indices].mean(dim=0)  # [Q]
        best_query_idx = int(tube_scores.argmax().item())
    print(f"[INFO] Selected best tube query index: {best_query_idx}")

    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for frame_idx in relevant_frame_indices:
        img = imags[frame_idx].copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size

        frame_boxes = pred_boxes[frame_idx]
        best_box_coords = frame_boxes[best_query_idx].detach().cpu().numpy()
        x_center, y_center, width, height = best_box_coords
        x1 = int((x_center - width / 2) * w)
        y1 = int((y_center - height / 2) * h)
        x2 = int((x_center + width / 2) * w)
        y2 = int((y_center + height / 2) * h)

        draw.rectangle((x1, y1, x2, y2), outline="red", width=5)
        save_path = os.path.join(output_dir, f"{base_name}_frame_{frame_idx:04d}.jpg")
        img.save(save_path)

        if do_attn:
            per_layer_vecs = []
            for layer_attn in attn_cache_raw:
                if isinstance(layer_attn, torch.Tensor) and layer_attn.dim() == 2:
                    if frame_idx < layer_attn.size(0):
                        per_layer_vecs.append(layer_attn[frame_idx:frame_idx + 1])

            if len(per_layer_vecs) > 0:
                _, blended_img = build_attn_heatmap(imags[frame_idx], per_layer_vecs, spatial_shapes, level_idx=2)
                if blended_img is not None:
                    attention_save_path = os.path.join(
                        output_dir,
                        f"{base_name}_frame_{frame_idx:04d}_attn_lvl2.jpg"
                    )
                    blended_img.save(attention_save_path)
                else:
                    print(f"[WARN] [frame {frame_idx}] build_attn_heatmap returned None.")
            else:
                print(f"[WARN] [frame {frame_idx}] No per-layer attention vectors; skip.")

    print(f"[INFO] Successfully processed and saved {len(relevant_frame_indices)} frames with bounding boxes.")


def run_text(model, processor, query, output_dir):
    modal = "text"
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": query},
            ]
        }
    ]

    inputs = processor(
        images=None,
        text=conversation,
        merge_size=1,
        return_tensors="pt",
    )

    output = mm_infer(
        inputs,
        model=model,
        tokenizer=processor.tokenizer,
        do_sample=False,
        modal=modal
    )
    print("Model Output:", output)
    save_text_output(output, output_dir, None, query)


# ====================== main ======================

def main():
    parser = argparse.ArgumentParser("DeViL unified demo for image / video / text")
    parser.add_argument("--modal", type=str, required=True,
                        choices=["image", "video", "text"],
                        help="Which modality to run: image / video / text")
    parser.add_argument("--model_path", type=str, default="/home/gaoshida/STVG/DeViL/weights/gdino_vllama_stage3_temporal_consistence/llm_lo_merge",
                        help="Path to model weights (llm_lo_merge)")
    parser.add_argument("--media_path", type=str, default=None,
                        help="Path to image or video. Not needed for text-only mode.")
    parser.add_argument("--query", type=str, required=True,
                        help="Text query / instruction.")
    parser.add_argument("--visualize_attention", action="store_true",
                        help="If set, will try to visualize attention maps.")
    parser.add_argument("--top_k", type=int, default=1,
                        help="Top-K boxes to draw for image mode.")
    parser.add_argument("--output_root", type=str,
                        default="/home/gaoshida/STVG/DeViL/assets/result",
                        help="Root directory to save all outputs.")
    args = parser.parse_args()

    disable_torch_init()
    model, processor = model_init(args.model_path)

    os.makedirs(args.output_root, exist_ok=True)

    if args.media_path is not None:
        base_name = os.path.splitext(os.path.basename(args.media_path))[0]
    else:
        base_name = "text_only"

    output_dir = os.path.join(args.output_root, base_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Outputs will be saved under: {output_dir}")

    if args.modal == "image":
        if args.media_path is None:
            raise ValueError("Image mode requires --media_path to be an image file.")
        run_image(
            model=model,
            processor=processor,
            image_path=args.media_path,
            query=args.query,
            output_dir=output_dir,
            visualize_attention=args.visualize_attention,
            top_k=args.top_k
        )
    elif args.modal == "video":
        if args.media_path is None:
            raise ValueError("Video mode requires --media_path to be a video file.")
        run_video(
            model=model,
            processor=processor,
            video_path=args.media_path,
            query=args.query,
            output_dir=output_dir,
            visualize_attention=args.visualize_attention
        )
    else:  # text
        run_text(
            model=model,
            processor=processor,
            query=args.query,
            output_dir=output_dir
        )


if __name__ == "__main__":
    main()
