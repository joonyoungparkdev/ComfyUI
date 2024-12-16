import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
import argparse


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def ltp_solution(input_image_path: str, output_folder_path: str):
    import_custom_nodes()
    with torch.inference_mode():
        checkpointloadersimple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        checkpointloadersimple_4 = checkpointloadersimple.load_checkpoint(
            ckpt_name="flat2DAnimerge_v45Sharp.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        loadimage_15 = loadimage.load_image(
            image=input_image_path
        )

        moondreamquery = NODE_CLASS_MAPPINGS["MoondreamQuery"]()
        moondreamquery_19 = moondreamquery.process(
            question="describe the picture in great detail",
            keep_model_loaded=True,
            model="moondream1",
            # model="moondream2",
            max_new_tokens=256,
            images=get_value_at_index(loadimage_15, 0),
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        cliptextencode_6 = cliptextencode.encode(
            text=get_value_at_index(moondreamquery_19, 0),
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_7 = cliptextencode.encode(
            text="(worst quality:0.8), embedding:verybadimagenegative_v1.3:1.0, (surreal:0.8), (modernism:0.8), (art deco:0.8), (art nouveau:0.8)",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        cliptextencode_21 = cliptextencode.encode(
            text="(best-quality:0.9), perfect anime illustration",
            clip=get_value_at_index(checkpointloadersimple_4, 1),
        )

        vaeencode = NODE_CLASS_MAPPINGS["VAEEncode"]()
        vaeencode_25 = vaeencode.encode(
            pixels=get_value_at_index(loadimage_15, 0),
            vae=get_value_at_index(checkpointloadersimple_4, 2),
        )

        controlnetloader = NODE_CLASS_MAPPINGS["ControlNetLoader"]()
        controlnetloader_27 = controlnetloader.load_controlnet(
            control_net_name="1.5/control_v11p_sd15_openpose_fp16.safetensors"
        )

        ipadapterunifiedloader = NODE_CLASS_MAPPINGS["IPAdapterUnifiedLoader"]()
        ipadapteradvanced = NODE_CLASS_MAPPINGS["IPAdapterAdvanced"]()
        conditioningconcat = NODE_CLASS_MAPPINGS["ConditioningConcat"]()
        dwpreprocessor = NODE_CLASS_MAPPINGS["DWPreprocessor"]()
        controlnetapply = NODE_CLASS_MAPPINGS["ControlNetApply"]()
        ksampler = NODE_CLASS_MAPPINGS["KSampler"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        saveimage = NODE_CLASS_MAPPINGS["SaveImage"]()

        for q in range(1):
            ipadapterunifiedloader_14 = ipadapterunifiedloader.load_models(
                preset="PLUS (high strength)",
                model=get_value_at_index(checkpointloadersimple_4, 0),
            )

            ipadapteradvanced_24 = ipadapteradvanced.apply_ipadapter(
                weight=1,
                weight_type="style and composition",
                combine_embeds="concat",
                start_at=0,
                end_at=1,
                embeds_scaling="K+V",
                model=get_value_at_index(ipadapterunifiedloader_14, 0),
                ipadapter=get_value_at_index(ipadapterunifiedloader_14, 1),
                image=get_value_at_index(loadimage_15, 0),
            )

            conditioningconcat_22 = conditioningconcat.concat(
                conditioning_to=get_value_at_index(cliptextencode_21, 0),
                conditioning_from=get_value_at_index(cliptextencode_6, 0),
            )

            dwpreprocessor_28 = dwpreprocessor.estimate_pose(
                detect_hand="enable",
                detect_body="enable",
                detect_face="enable",
                resolution=512,
                bbox_detector="yolox_l.onnx",
                pose_estimator="dw-ll_ucoco_384_bs5.torchscript.pt",
                scale_stick_for_xinsr_cn="disable",
                image=get_value_at_index(loadimage_15, 0),
            )

            controlnetapply_26 = controlnetapply.apply_controlnet(
                strength=0.5,
                conditioning=get_value_at_index(conditioningconcat_22, 0),
                control_net=get_value_at_index(controlnetloader_27, 0),
                image=get_value_at_index(dwpreprocessor_28, 0),
            )

            ksampler_3 = ksampler.sample(
                seed=random.randint(1, 2**64),
                steps=60,
                cfg=10,
                sampler_name="dpmpp_2m",
                scheduler="karras",
                denoise=0.37,
                model=get_value_at_index(ipadapteradvanced_24, 0),
                positive=get_value_at_index(controlnetapply_26, 0),
                negative=get_value_at_index(cliptextencode_7, 0),
                latent_image=get_value_at_index(vaeencode_25, 0),
            )

            vaedecode_8 = vaedecode.decode(
                samples=get_value_at_index(ksampler_3, 0),
                vae=get_value_at_index(checkpointloadersimple_4, 2),
            )

            saveimage_30, full_output_path = saveimage.save_images(
                filename_prefix="lenticular_toon_",
                images=get_value_at_index(vaedecode_8, 0),
                output_path=output_folder_path,
            )
    return full_output_path

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    # Add arguments
    parser.add_argument('--path', type=str, help='Path of input image.', required=True)     # Custom arg for lenticular_toon_protraits
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    output_folder = r""
    output_img_path = ltp_solution(args.path, output_folder)
    print(output_img_path)
