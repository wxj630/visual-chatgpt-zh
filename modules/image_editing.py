from modules.utils import *

class ImageEditing:
    def __init__(self, device, pretrained_model_dir):
        print("Initializing ImageEditing to %s" % device)
        self.device = device
        self.mask_former = MaskFormer(device=self.device, pretrained_model_dir=pretrained_model_dir)
        self.revision = 'fp16' if 'cuda' in device else None
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            f"{pretrained_model_dir}/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype).to(device)

    @prompts(name="Remove Something From The Photo",
             description="useful when you want to remove and object or something from the photo "
                         "from its description or location. "
                         "The input to this tool should be a comma seperated string of two, "
                         "representing the image_path and the object need to be removed. ")
    def inference_remove(self, inputs):
        image_path, to_be_removed_txt = inputs.split(",")
        return self.inference_replace(f"{image_path},{to_be_removed_txt},background")

    @prompts(name="Replace Something From The Photo",
             description="useful when you want to replace an object from the object description or "
                         "location with another object from its description. "
                         "The input to this tool should be a comma seperated string of three, "
                         "representing the image_path, the object to be replaced, the object to be replaced with ")
    def inference_replace(self, inputs):
        image_path, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        original_image = Image.open(image_path)
        original_size = original_image.size
        mask_image = self.mask_former.inference(image_path, to_be_replaced_txt)
        updated_image = self.inpaint(prompt=replace_with_txt, image=original_image.resize((512, 512)),
                                     mask_image=mask_image.resize((512, 512))).images[0]
        updated_image_path = get_new_image_name(image_path, func_name="replace-something")
        updated_image = updated_image.resize(original_size)
        updated_image.save(updated_image_path)
        print(
            f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
            f"Output Image: {updated_image_path}")
        return updated_image_path