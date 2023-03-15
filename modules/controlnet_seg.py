from modules.utils import *

class Image2Seg:
    def __init__(self, device, pretrained_model_dir):
        print("Initializing Image2Seg")
        self.image_processor = AutoImageProcessor.from_pretrained(f"{pretrained_model_dir}/upernet-convnext-small")
        self.image_segmentor = UperNetForSemanticSegmentation.from_pretrained(f"{pretrained_model_dir}/upernet-convnext-small")
        self.ade_palette = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                            [102, 255, 0], [92, 0, 255]]

    @prompts(name="Segmentation On Image",
             description="useful when you want to detect segmentations of the image. "
                         "like: segment this image, or generate segmentations on this image, "
                         "or peform segmentation on this image. "
                         "The input to this tool should be a string, representing the image_path")
    def inference(self, inputs):
        image = Image.open(inputs)
        pixel_values = self.image_processor(image, return_tensors="pt").pixel_values
        with torch.no_grad():
            outputs = self.image_segmentor(pixel_values)
        seg = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)  # height, width, 3
        palette = np.array(self.ade_palette)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg.astype(np.uint8)
        segmentation = Image.fromarray(color_seg)
        updated_image_path = get_new_image_name(inputs, func_name="segmentation")
        segmentation.save(updated_image_path)
        print(f"\nProcessed Image2Pose, Input Image: {inputs}, Output Pose: {updated_image_path}")
        return updated_image_path


class SegText2Image:
    def __init__(self, device, pretrained_model_dir):
        print("Initializing SegText2Image to %s" % device)
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.controlnet = ControlNetModel.from_pretrained(f"{pretrained_model_dir}/sd-controlnet-seg",
                                                          torch_dtype=self.torch_dtype)
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            f"{pretrained_model_dir}/stable-diffusion-v1-5", controlnet=self.controlnet, safety_checker=None,
            torch_dtype=self.torch_dtype)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to(device)
        self.seed = -1
        self.a_prompt = 'best quality, extremely detailed'
        self.n_prompt = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit,' \
                        ' fewer digits, cropped, worst quality, low quality'

    @prompts(name="Generate Image Condition On Segmentations",
             description="useful when you want to generate a new real image from both the user desciption and segmentations. "
                         "like: generate a real image of a object or something from this segmentation image, "
                         "or generate a new real image of a object or something from these segmentations. "
                         "The input to this tool should be a comma seperated string of two, "
                         "representing the image_path and the user description")
    def inference(self, inputs):
        image_path, instruct_text = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        image = Image.open(image_path)
        self.seed = random.randint(0, 65535)
        seed_everything(self.seed)
        prompt = instruct_text + ', ' + self.a_prompt
        image = self.pipe(prompt, image, num_inference_steps=20, eta=0.0, negative_prompt=self.n_prompt,
                          guidance_scale=9.0).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="segment2image")
        image.save(updated_image_path)
        print(f"\nProcessed SegText2Image, Input Seg: {image_path}, Input Text: {instruct_text}, "
              f"Output Image: {updated_image_path}")
        return updated_image_path