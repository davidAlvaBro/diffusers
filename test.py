import numpy as np 
import torch
import cv2
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
from diffusers.utils import load_image

device = "cuda"

# 1) Load two controlnets: depth + pose
cn_depth = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-depth",
    torch_dtype=torch.float16,
)

cn_pose = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=[cn_depth, cn_pose],      # <-- multiple controlnets
    torch_dtype=torch.float16
).to(device)

# 2) Prepare images
# depth_image = np.load("depth.npy")
# depth_image = depth_image[462:462+403, 305:305+478]
# depth_image = cv2.resize(depth_image, (512, 448))
pose_image  = load_image("pose.png")
depth_image  = load_image("depth.png")

prompt = "Two men with black hair in gray suits facing the same way, standing slightly apart, located on an empty street, muted colors, single everyday image, this photo is part of collection where these people are being photographed from all angles"
n_prompt = "extra fingers, too few fingers, bad quality, worst quality"

# 3) Call with *lists* for both images and scales
images = pipe(
    prompt=prompt,
    num_inference_steps=30,
    image=[depth_image],#, pose_image],
    controlnet_conditioning_scale=[0.5, 0.8], # depth weight, pose weight
    negative_prompt=n_prompt
).images

images[0].save("out.png")
