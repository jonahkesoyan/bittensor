import bittensor as bt

# from bytes to PIL image
from PIL import Image
from io import BytesIO
import numpy as np
import requests
import base64

hotkey = 'asdasd'

text = """Realistic, Photography, human-like, sony alpha, lossless quality, realistic beautiful witch, ornate rococo styled skimpwear robe, fit and attractive, short beautiful voluminous hair with huge soft chest, alluring pose in front of her fans, full body shot photo composition, beautiful legs and armor, hyper realistic photography, Masterpiece, superrealism, realistic face, realistic hair, realistic eyes, realistic characters, realistic environment, realistic body, realistic physiology, realistic detailed, stunning realistic photo of a realistic dramatic character, hight quality, best quality, fusion between jeremy mann and childe hassam and daniel f gerhartz and rosa bonheur and thomas eakins, by lucian freud and candido portinari and charlie bowater, by wes anderson, Cinematic, smooth skin, flawless complexion, uplight, illuminating, nice shot, fine detail, CinemaHelper, PhotoHelper, 16K, gfpgan, trending on pexels, fullbody"""


# open image from path and be PIL.Image
image = Image.open('/home/carro/moebius.png')

buffered = BytesIO()
image.save(buffered, format="PNG")  # You can use "PNG" if you prefer

# Encode image buffer to base64
image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
data = { 
  "text": text,
  # "image": image_base64,
  "height": 768, # anything less than 512x512 causes image degradation
  "width": 512,
  "timeout": 120,
  "num_images_per_prompt": 1,
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "negative_prompt": ""
}

# from bytes to PIL image
req = requests.post('http://127.0.0.1:8092/TextToImage/Forward/?hotkey={}'.format(hotkey), json=data)

img_base64 = req.text

# Decode base64 string into bytes
img_bytes = base64.b64decode(img_base64)

# Load bytes into a PIL image
pil = Image.open(BytesIO(img_bytes))

pil.save('test.png')
print('saved image to test.png')