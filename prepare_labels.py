
import math
from pathlib import Path
import json
import sys
from typing import Tuple

import cv2
import numpy as np

import numpy.core as npcore
sys.modules.setdefault('numpy._core', npcore)
# Common submodules that may appear in pickles:
if hasattr(npcore, '_multiarray_umath'):
    sys.modules['numpy._core._multiarray_umath'] = npcore._multiarray_umath
if hasattr(npcore, 'multiarray'):
    sys.modules['numpy._core.multiarray'] = npcore.multiarray

def draw_bodypose(canvas, candidate, subset):
    stickwidth = 4
    limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18], [3, 17], [6, 18]]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
    for i in range(18):
        for n in range(len(subset)):
            index = int(subset[n][i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), 4, colors[i], thickness=-1)
    for i in range(17):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i]) - 1]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
    return canvas

def resize_annotation(candidate, resolution, orig_shape):
    # Resize the annotation so that the maximal edge gets the resolution "resolution"
    H, W = orig_shape
    if type(resolution) == tuple: 
        # Check if resolution is divisible with 64 
        assert resolution[0] % 64 == 0 and resolution[1] % 64 == 0 
        H_new = resolution[0] 
        W_new = resolution[1]
    else: 
        H_new = int(np.round((H * float(resolution) / max(H, W)) / 64.0)) * 64 # TODO : could be changed to (H * resolution[0] // min(H, W) 
        W_new = int(np.round((W * float(resolution) / max(H, W)) / 64.0)) * 64 # TODO check that his is actually correct
    candidate[:, 1] = candidate[:, 1] * H_new / H
    candidate[:, 0] = candidate[:, 0] * W_new / W 
    return candidate, (H_new, W_new)


def get_annotations(annotations_path: Path, working_dir: Path, frames: dict, resolution: int = 512, zoom_in: bool = True) -> Tuple[np.ndarray, dict]: 
    """
    Loads all annotations, produces the annotation map and corrects camera intrinsics if 'zoom_in' is on
    """
    # minimum padding should be in some config
    min_pad = 50
    # Ensure that annotations folder exists 
    (working_dir / "annotations").mkdir(parents=True, exist_ok=True)

    # Load raw annotations
    annotations = np.load(annotations_path / "gt_annotation.npz", allow_pickle=True)
    candidate_full = annotations["candidate"]
    subset_full = annotations["subset"].astype(int)
    image_to_people = annotations["image_to_people"].item()

    # Get annotations per frame
    frame_dict = {}
    max_dif_x = 0
    max_dif_y = 0
    for frame in frames:
        # annotation for this frame 
        img_name = frame["file_path"]
        people = image_to_people[img_name]
        subset = np.array([subset_full[person] for person in people])
        candidates = candidate_full[subset.flatten()]
        _, inv = np.unique(subset.flatten(), return_inverse=True)
        subset = inv.reshape(subset.shape)

        # Also store where the annotation is in the image 
        # First entry is max x (width), second max y (height) 
        max_xy = candidates[:, :2].max(axis=0)
        min_xy = candidates[:, :2].min(axis=0)

        frame_dict[img_name] = {"candidates": candidates, 
                                "subset": subset,
                                "max": max_xy, 
                                "min": min_xy,
                                "origin_x": 0, 
                                "origin_y": 0,
                                "h": frame["h"],
                                "w": frame["w"], 
                                "frame": frame}
        dif = max_xy - min_xy 
        if dif[0] > max_dif_x: max_dif_x = int(dif[0]) 
        if dif[1] > max_dif_y: max_dif_y = int(dif[1])
    
    # Controlnet has difficulty generating people that are small - far away 
    if zoom_in: # TODO : crop_shape needs to be somewhere else too 
        crop_shape = np.array([max_dif_x + min_pad , max_dif_y + min_pad])
        # TODO : check that this updates the big dict
        for _, value in frame_dict.items():
            total_padding = crop_shape - (value["max"] - value["min"]) 
            value["candidates"][:,:2] = value["candidates"][:,:2] - value["min"] + total_padding/2
            value["origin_x"], value["origin_y"] = value["min"] - total_padding/2
            value["h"] = crop_shape[1]
            value["w"] = crop_shape[0]

    # For each entry resize it, draw the corresponding canvas, and recalibrate the camera
    for key, value in frame_dict.items(): 
        value["candidates"], (new_h, new_w) = resize_annotation(value["candidates"], resolution=resolution, orig_shape=(value["h"], value["w"]))
        rescale_x = new_w/value["w"]
        rescale_y = new_h/value["h"]
        
        # Draw the annotation 
        canvas = np.zeros((new_h, new_w, 3)) 
        canvas = draw_bodypose(canvas=canvas, candidate=value["candidates"], subset=value["subset"])
        # Store the annotation
        anno_path = Path("annotations") / Path(value["frame"]["file_path"]).name
        cv2.imwrite(working_dir / anno_path, canvas) 

        # Make the corresponding depth map if the frame has it 
        if "depth_path" in value["frame"]: 
            depth = np.load(annotations_path / value["frame"]["depth_path"])
            depth = depth[value["origin_y"]:value["origin_y"] + crop_shape[1], value["origin_x"]:value["origin_x"] + crop_shape[0]]
            depth = cv2.resize(depth, (new_w, new_h))
            (working_dir / value["frame"]["depth_path"]).parent.mkdir(parents=True, exit_ok=True)
            cv2.imwrite(working_dir / value["frame"]["depth_path"], depth) 

        value["frame"]["h"] = new_h
        value["frame"]["w"] = new_w
        value["frame"]["fl_x"] = value["frame"]["fl_x"]*rescale_x
        value["frame"]["fl_y"] = value["frame"]["fl_y"]*rescale_y
        value["frame"]["cx"] = (value["frame"]["cx"] - value["origin_x"])*rescale_x
        value["frame"]["cy"] = (value["frame"]["cy"] - value["origin_y"])*rescale_y
        value["frame"]["annotation_path"] = str(anno_path)
    
    return frames


# def get_annotations(annotations_path: Path, cameras_path: Path, output_path: Path, resolution: int=512): # TODO update description when method is done 
#     """
#     Loads all poses for a scene, constructs the annotation canvas for each view, 
#     resize each pose and update the transform.json to fit this.

#     returns numpy array of annotation canvases that ControlNet can use. 
#     """
#     # TODO maybe do some assertions? 
#     output_path.mkdir(parents=True, exist_ok=True)

#     # Get the annotations
#     annotations = np.load(annotations_path, allow_pickle=True)
#     candidate_full = annotations["candidate"]
#     subset_full = annotations["subset"].astype(int)
#     image_to_people = annotations["image_to_people"].item()
#     img_names = [Path(p).name for p in image_to_people.keys()]

#     n_images = len(image_to_people)
#     # Stores the (min_x, min_y, max_x, max_y, scale_h, scale_w) annotation value for each image
#     views = np.zeros((n_images, 6)) # The scale goes from the old view to the new
#     h_min = 0 # the smallest height can be to incoperate all annotation views 
#     w_min = 0 
#     list_of_canvases = []
#     for i, person_key in enumerate(image_to_people): 
#         # Now looking at one image/view
#         people = image_to_people[person_key]
#         subset = subset_full[people].flatten()
#         # "Zoom in" and resize to get something %64 = 0 because the VAE encoder needs it 
#         candidate_full[subset], (height, width, _), (min_x, min_y, max_x, max_y) = focus_on_annotation(candidate=candidate_full[subset])
#         candidate_full[subset], (H, W) = resize_annotation(candidate=candidate_full[subset], resolution=resolution, orig_shape=(height, width)) # Fix views for resizing 
#         scale_h = H/height # The scale that goes from the "old" views to the new 
#         scale_w = W/width 
#         views[i] = np.array([min_x*scale_w, min_y*scale_h, max_x*scale_w, max_y*scale_h, scale_h, scale_w]) # TODO make sure this fits with the shape of the new image somehow? 
#         if h_min < H : # TODO replace this with a "resolution" and just let it be constant. then the images are always square
#             h_min = H 
#         if w_min < W : 
#             w_min = W 
#         # Draw the annotation 
#         canvas = np.zeros((H, W, 3)) 
#         subset = subset_full[people]
#         canvas = draw_bodypose(canvas=canvas, candidate=candidate_full, subset=subset)
#         list_of_canvases.append(canvas)
    
#     # Put the annotation images into to canvas placeholder 
#     canvases = np.zeros((n_images, h_min, w_min, 3)) 
#     difs = np.zeros((n_images, 2)) 
#     for i, person_key in enumerate(image_to_people): 
#         # put the image into the middle of the final canvas 
#         (min_x, min_y, max_x, max_y, scale_h, scale_w) = views[i]
#         img_shape = list_of_canvases[i].shape
#         dif_h = int((h_min - img_shape[0])/2) 
#         dif_w = int((w_min - img_shape[1])/2)
#         difs[i] = np.array([dif_h, dif_w])
#         canvases[i, dif_h:dif_h + img_shape[0], dif_w:dif_w + img_shape[1]] = list_of_canvases[i] 
#         # Update the annotation if it is needed for debugging 
#         people = image_to_people[person_key]
#         subset = subset_full[people].flatten()
#         candidate_full[subset] = candidate_full[subset] + np.array([dif_w, dif_h, 0, 0]) # TODO save this annotation file somewhere too 
    
#     # Save updated cameras that are "zoomed in" 
#     with open(cameras_path, 'r') as file:
#         cameras_dict = json.load(file)
#     for frame in cameras_dict["frames"]: 
#         i = img_names.index(Path(frame["file_path"]).name) 
#         (min_x, min_y, max_x, max_y, scale_h, scale_w) = views[i]
#         dif_h, dif_w = difs[i]
#         # update data 
#         frame["fl_x"] = frame["fl_x"]*scale_w
#         frame["fl_y"] = frame["fl_y"]*scale_h
#         frame["cx"] = (frame["cx"]*scale_w - min_x) + dif_w # The second crop adds stuff
#         frame["cy"] = (frame["cy"]*scale_h - min_y) + dif_h
#         frame["w"] = w_min
#         frame["h"] = h_min
#     with open(output_path / "transforms.json", "w", encoding="utf-8") as file:
#         json.dump(cameras_dict, file, ensure_ascii=False, indent=2)
    
#     return canvases, img_names


if __name__ == "__main__":
    anno_path = Path("test/gt_annotation.npz")
    cameras_path = Path("test/transforms.json")
    out_path = Path("test/out")

    get_annotations(annotations_path=anno_path, cameras_path=cameras_path, output_path=out_path)