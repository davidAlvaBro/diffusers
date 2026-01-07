import argparse
from pathlib import Path
import json

import numpy as np 
import torch
import cv2
from PIL import Image

from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image


# def run_controlnet(pose_condition: Path, gen_path: Path, prompt: str, depth_path: Path | None = None): 
def run_controlnet(pose_condition: Image, pose_condition_zoomed: Image, gen_path: Path, prompt: str, reference_frame, depth_path: Path | None = None): 
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
    # prompt = "Two men with black hair in gray suits facing the same way, standing slightly apart, located on an empty street, muted colors, single everyday image, this photo is part of collection where these people are being photographed from all angles"
    n_prompt = "extra fingers, too few fingers, bad quality, worst quality, multiple stitched together images"

    # 3) Call with *lists* for both images and scales
    images = pipe(
        prompt=prompt,
        num_inference_steps=50, # TODO also in config 
        image=conditions,
        controlnet_conditioning_scale=weights,
        negative_prompt=n_prompt
    ).images

    images[0].save(gen_path.parent / "full.png")

    # Create crop 
    gen_img = np.array(images[0]).astype(np.float32) / 255.0 
    crop = gen_img[reference_frame["crop_y_min"]:reference_frame["crop_y_max"], reference_frame["crop_x_min"]:reference_frame["crop_x_max"]]
    crop = cv2.resize(crop, (pose_condition_zoomed.size))
    
    conditions = [pose_condition_zoomed]

    images = pipe(
        prompt=prompt,
        num_inference_steps=50, # TODO also in config 
        image=conditions,
        controlnet_conditioning_scale=weights,
        negative_prompt=n_prompt, 
        start_step_idx=25,
        inpainting_img=torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device)
    ).images

    images[0].save(gen_path.parent / "crop.png")

    gen_crop = np.array(images[0]).astype(np.float32) / 255.0 
    gen_crop = cv2.resize(gen_crop, (reference_frame["crop_x_max"] - reference_frame["crop_x_min"], reference_frame["crop_y_max"]- reference_frame["crop_y_min"]))

    gen_img[reference_frame["crop_y_min"]:reference_frame["crop_y_max"], reference_frame["crop_x_min"]:reference_frame["crop_x_max"]] = gen_crop

    Image.fromarray((gen_img * 255.0).clip(0, 255).astype(np.uint8), mode="RGB").save(gen_path)


def preprocess_two_views(data_dir, ref_frame): 
    # TODO - fix this so that it also works for tall images
    # First make the wide frame square 
    # max_side_length = max(ref_frame["2"], ref_frame["fl_y"])
    # min_side_length = min(ref_frame["fl_x"], ref_frame["fl_y"])
    # dif = (max_side_length - min_side_length) // 2
    dif = (ref_frame["w"] - ref_frame["h"]) // 2
    wide_annotation = load_image(str(data_dir / ref_frame["annotation_path"]))
    wide_annotation = wide_annotation.crop((dif, 0, ref_frame["h"] + dif, ref_frame["h"]))
    ref_frame["cx"] = ref_frame["cx"] - dif
    ref_frame["w"] = ref_frame["w"] - 2*dif

    # Adjust what this means for the crop  
    ref_frame["crop_x_min"] = ref_frame["crop_x_min"] - dif 
    ref_frame["crop_x_max"] = ref_frame["crop_x_max"] - dif

    # Now resize to (512, 512)
    final_shape = (512, 512)
    wide_annotation = wide_annotation.resize(final_shape, Image.BICUBIC)
    scale = final_shape[0] / ref_frame["w"]
    ref_frame["fl_x"] = ref_frame["fl_x"]*scale
    ref_frame["fl_y"] = ref_frame["fl_y"]*scale
    ref_frame["cx"] = ref_frame["cx"]*scale
    ref_frame["cy"] = ref_frame["cy"]*scale
    ref_frame["h"] = ref_frame["h"]*scale
    ref_frame["w"] = ref_frame["w"]*scale

    # Again adjust for the other camera 
    ref_frame["crop_x_min"] = int(np.round(ref_frame["crop_x_min"]*scale))
    ref_frame["crop_x_max"] = int(np.round(ref_frame["crop_x_max"]*scale)) 
    ref_frame["crop_y_min"] = int(np.round(ref_frame["crop_y_min"]*scale))
    ref_frame["crop_y_max"] = int(np.round(ref_frame["crop_y_max"]*scale)) 

    # Load the narrow annotation
    narrow_annotation = load_image(str(data_dir / ref_frame["zoomed_annotation_path"]))

    return wide_annotation, narrow_annotation, ref_frame


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="Apply pose and depth controlnet on the reference image and save it for lifting pipeline.")
    parser.add_argument("--data-dir", default="../data/testy", type=str, help="Path to folder where images will be stored in the folder 'images'.")
    parser.add_argument("--prompt", default=" ", type=str, help="Prompt used for image generation.")
    args = parser.parse_args()
    data_dir = Path(args.data_dir) 

    # Paths to annotation, metadata, output metadata and output image folders
    metadata_path = data_dir / "transforms.json"
    out_json_path = data_dir / "controlnet/transforms.json"
    out_imgs_path = data_dir / "controlnet"
    prompt_path = data_dir / "prompt.txt"
    
    # Save the prompt for logging purposes 
    if args.prompt == " ": 
        prompt = open(prompt_path).read()
    else : 
        prompt = args.prompt
        with open(prompt_path, "w") as f:
            f.write(prompt)

    # Get metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    trajectory = metadata["trajectory"]
    reference_frame_idx = metadata["trajectory_ref"]
    reference_frame = trajectory[reference_frame_idx]

    wide, narrow, reference_frame = preprocess_two_views(data_dir=data_dir, ref_frame=reference_frame)

    # annotation_path_zoomed = data_dir / reference_frame["zoomed_annotation_path"]
    # annotation_path = data_dir / reference_frame["annotation_path"]
    generated_path = out_imgs_path / trajectory[reference_frame_idx]["file_path"]
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    # # Resize depth 
    # (h,w) = reference_frame["zoomed_h"], reference_frame["zoomed_w"]
    # x_min, x_max = reference_frame["crop_x_min"], reference_frame["crop_x_max"]
    # y_min, y_max = reference_frame["crop_y_min"], reference_frame["crop_y_max"]
    # depth = np.load(data_dir / reference_frame["depth_path"])

    # depth_cropped = depth[y_min:y_max, x_min:x_max]
    # depth_resized = cv2.resize(depth_cropped, (w, h))
    
    # depth_path = out_imgs_path / reference_frame["depth_path"]
    # depth_path = Path(str(depth_path)[:-4] + ".png")
    # depth_path.parent.mkdir(parents=True, exist_ok=True)
    # # print(depth_path, type(depth_resized), depth_resized.shape)
    # cv2.imwrite(depth_path, depth_resized)
    
    
    # Controlnet 
    run_controlnet(pose_condition=wide, pose_condition_zoomed=narrow, gen_path=generated_path, prompt=prompt, reference_frame=reference_frame)#, depth_path=depth_path)
    # run_controlnet(pose_condition=annotation_path, pose_condition_zoomed=annotation_path_zoomed, gen_path=generated_path, prompt=prompt, reference_frame=reference_frame)#, depth_path=depth_path)

    # Nothing builds on 'frames' after the controlnet pipeline 
    metadata["frames"] = metadata.pop("trajectory")
    metadata["ref"] = metadata.pop("trajectory_ref")
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)   

    