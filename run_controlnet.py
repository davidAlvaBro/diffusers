import argparse
from pathlib import Path
import json

import numpy as np 
import torch
import cv2

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

from prepare_labels import get_annotations


def run_controlnet(pose_condition: Path, gen_path: Path, depth_path: Path | None = None): 
    """
    Given a path to an annotation dataset, a path to the camera parameters, and an output path 
    generate new pose conditioned images from each of these camera views. 
    """
    device = 'cuda'
    controlnets = [] 
    conditions = []
    weights = []

    # Pose control
    cn_pose = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=torch.float16,
    )
    pose_condition = load_image(pose_condition)
    conditions += [pose_condition]
    controlnets += [cn_pose]
    weights += [0.8] # TODO put this hardcoded value somewhere else 

    if depth_path is not None: 
        cn_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        )
        depth_condition = load_image(depth_path)
        conditions += [depth_condition]
        controlnets += [cn_depth]
        weights += [0.4] # TODO put this hardcoded value somewhere else 

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnets,
        torch_dtype=torch.float16
    ).to(device)

    # TODO also put this in some configuration file 
    prompt = "Two men with black hair in gray suits facing the same way, standing slightly apart, located on an empty street, muted colors, single everyday image, this photo is part of collection where these people are being photographed from all angles"
    n_prompt = "extra fingers, too few fingers, bad quality, worst quality"

    # 3) Call with *lists* for both images and scales
    images = pipe(
        prompt=prompt,
        num_inference_steps=50, # TODO also in config 
        image=conditions,
        controlnet_conditioning_scale=weights,
        negative_prompt=n_prompt
    ).images

    images[0].save(gen_path)


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Apply pose and depth controlnet on the reference image and save it for lifting pipeline.")
    parser.add_argument("--data-dir", default="/data/test", type=str, help="Path to folder where images will be stored in the folder 'images'.")
    parser.add_argument("--adjust-all", action="store_true", help="If all frames needs to zoom in on the annotation.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) 

    # Paths to annotation, metadata, output metadata and output image folders
    anno_path = data_dir / "gt_annotation.npz"
    metadata_path = data_dir / "transforms.json"
    out_json_path = data_dir / "controlnet/transforms.json"
    out_imgs_path = data_dir / "controlnet"

    # Get metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    trajectory = metadata["trajectory"]
    reference_frame_idx = metadata["trajectory_ref"]

    # Prepare annotations - resize camera views 
    trajectory = get_annotations(annotations_path=anno_path, working_dir=out_imgs_path, frames=trajectory, resolution=512, zoom_in=True)
    annotation  = cv2.imread(str(out_imgs_path / trajectory[reference_frame_idx]["annotation_path"]))
    generated_path = out_imgs_path / trajectory[reference_frame_idx]["file_path"]
    generated_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Controlnet 
    run_controlnet(condition=annotation, gen_path=generated_path)

    # Save metadata 
    metadata["frames"] = metadata.pop("trajectory")
    metadata["ref"] = metadata.pop("trajectory_ref")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


    