# import json
# import torch
# import requests
# from io import BytesIO
# import bittensor as bt
# import soundfile as sf
# import base64
# bt.trace()

# hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'

# pcm_audio, fs = sf.read('/home/jason/assets/urgency.wav')

# # Convert audio to bytes and write to buffer
# audio_buffer = BytesIO()
# sf.write( audio_buffer, pcm_audio.numpy().squeeze(), 22050, format='WAV' )

# # Convert buffer to base64 string
# audio_base64 = base64.b64encode( audio_buffer.getvalue() ).decode('utf-8')

# data = {
#   "speech": audio_base64,
#   "timeout": 12,
# }

# req = requests.post('http://127.0.0.1:8092/SpeechToText/Forward/?hotkey={}'.format(hotkey), json=data)
# print(req)
# print(req.text)

# x = json.loads(req.text)
# emb = torch.Tensor(x)
# print("emb.shape: ", emb.shape)


import requests
import bittensor as bt
from io import BytesIO
import soundfile as sf
import base64
from datasets import load_dataset

bt.trace()


def get_audio_base64_example():
    ds = load_dataset( "hf-internal-testing/librispeech_asr_demo", "clean", split="validation" )
    audio_array = ds[0]["audio"]["array"]
    samplerate = ds[0]["audio"]["sampling_rate"]
    audio_buffer = BytesIO()
    sf.write( audio_buffer, audio_array, samplerate, format='WAV' )
    vibes = audio_buffer.getvalue()
    audio_base64 = base64.b64encode( vibes ).decode( 'utf-8' )
    return audio_base64, vibes


hotkey = '5C86aJ2uQawR6P6veaJQXNK9HaWh6NMbUhTiLs65kq4ZW3NH'
audio_64, audio_bytes = get_audio_base64_example()

data = {
  "speech": audio_64,
  "timeout": 12,
}
req = requests.post('http://127.0.0.1:8092/SpeechToText/Forward/?hotkey={}'.format(hotkey), json=data)
print(req)
print(req.text)
