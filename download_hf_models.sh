# huggingface下载好的模型路径，可以修改到自己想要存放模型的路径
YOUR_HF_DOWNLOAD_PATH={your_hf_models_path}

cd $YOUR_HF_DOWNLOAD_PATH
nohup git clone https://huggingface.co/Salesforce/blip-image-captioning-base &
nohup git clone https://huggingface.co/runwayml/stable-diffusion-v1-5 &
nohup git clone https://huggingface.co/runwayml/stable-diffusion-inpainting &
nohup git clone https://huggingface.co/CIDAS/clipseg-rd64-refined &
nohup git clone https://huggingface.co/timbrooks/instruct-pix2pix &
nohup git clone https://huggingface.co/Salesforce/blip-vqa-base &

nohup git clone https://huggingface.co/lllyasviel/ControlNet
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-canny &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-seg &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-scribble &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-normal &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-mlsd &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-depth &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-hed &
nohup git clone https://huggingface.co/lllyasviel/sd-controlnet-openpose &
nohup git clone https://huggingface.co/openmmlab/upernet-convnext-small &
nohup git clone https://huggingface.co/Intel/dpt-hybrid-midas