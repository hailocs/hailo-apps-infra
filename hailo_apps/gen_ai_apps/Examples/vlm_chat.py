from hailo_platform import VDevice
from hailo_platform.genai import VLM
import cv2

vdevice = VDevice()
vlm = VLM(vdevice, 'Qwen2-VL-2B-Instruct.hef')
prompt = [
    {
        "role": "system", 
        "content": [{"type": "text", "text": 'You are a helpful assistant that analyzes images and answers questions about them.'}]
    },
    {
        "role": "user", 
        "content": [
            {"type": "image"}, 
            {"type": "text", "text": 'Please describe the image.'}
        ]
    }
]
print(vlm.generate_all(prompt=prompt, frames=[cv2.cvtColor(cv2.imread('/home/michaelf/Downloads/vlm_demo/barcode-example.png'), cv2.COLOR_BGR2RGB)], temperature=0.1, seed=42, max_generated_tokens=200).split(". [{'type'")[0])