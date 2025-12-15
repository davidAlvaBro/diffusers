import argparse
from pathlib import Path
import json

import numpy as np 
import torch
import cv2

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image


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
    pose_condition = load_image(str(pose_condition))
    conditions += [pose_condition]
    controlnets += [cn_pose]
    weights += [0.8] # TODO put this hardcoded value somewhere else 

    if depth_path is not None: 
        cn_depth = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        )
        depth_condition = load_image(str(depth_path))
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
    n_prompt = "extra fingers, too few fingers, bad quality, worst quality, multiple stitched together images"

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
    metadata_path = data_dir / "transforms.json"
    out_json_path = data_dir / "controlnet/transforms.json"
    out_imgs_path = data_dir / "controlnet"

    # Get metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    trajectory = metadata["trajectory"]
    reference_frame_idx = metadata["trajectory_ref"]
    reference_frame = trajectory[reference_frame_idx]

    annotation_path = data_dir / reference_frame["zoomed_annotation_path"]
    generated_path = out_imgs_path / trajectory[reference_frame_idx]["file_path"]

    # Resize depth 
    (h,w) = reference_frame["zoomed_h"], reference_frame["zoomed_w"]
    x_min, x_max = reference_frame["zoomed_x_min"], reference_frame["zoomed_x_max"]
    y_min, y_max = reference_frame["zoomed_y_min"], reference_frame["zoomed_y_max"]
    depth = np.load(data_dir / reference_frame["depth_path"])

    depth_cropped = depth[y_min:y_max, x_min:x_max]
    depth_resized = cv2.resize(depth_cropped, (w, h))
    
    depth_path = out_imgs_path / reference_frame["depth_path"]
    depth_path.parent.mkdir(parents=True, exists=True)
    cv2.imwrite(depth_path, depth_resized)
    
    # Controlnet 
    run_controlnet(pose_condition=annotation_path, gen_path=generated_path, depth_path=depth_path)

    