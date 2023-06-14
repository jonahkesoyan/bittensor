# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import torch
import argparse
import bittensor
import base64
from io import BytesIO

from PIL import Image
from audiocraft.models import musicgen
import soundfile as sf
from audiocraft.data.audio import audio_write

from audiocraft.utils.notebook import display_audio
from typing import List, Dict, Union, Tuple, Optional

def config():       
    parser = argparse.ArgumentParser( description = 'MusicGen text to music Miner' )
    parser.add_argument( '--neuron.model_name', type=str, help='Name of the model to use.', default = "medium" )
    parser.add_argument( '--device', type=str, help='Device to load model', default="cuda" )
    bittensor.base_miner_neuron.add_args( parser )
    return bittensor.config( parser )

def main( config ):
    bittensor.trace()

    # --- Build the base miner
    base_miner = bittensor.base_miner_neuron( netuid = 17, config = config )

    # --- Build diffusion pipeline ---
    # lpw_stable_diffusion is used to increase CLIP token length from 77 only works for text2img
    model = musicgen.MusicGen.get_pretrained(config.neuron.model_name, device=config.device)

    # --- Build Synapse ---
    class MusicGen( bittensor.TextToMusicSynapse ):

        def priority( self, forward_call: "bittensor.SynapseCall" ) -> float: 
            # return base_miner.priority( forward_call )
            return 0.0

        def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
            # return base_miner.blacklist( forward_call )
            return False
        
        def forward( self, text: str, sample: str, duration: int, ) -> List[str]:
            model.set_generation_params(duration=duration)

            output = model.generate(text, progress=True)
            print(output)
            for idx, one_wav in enumerate(output):
                audio_write(f'tmp_{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)
                # strategy = clip, peak or rms

            
            audio_buffer = BytesIO()
            with open(f'tmp_20.wav', 'rb') as f:
                content = f.read()
                sf.write(audio_buffer, content, model.sample_rate, format='wav')
                vibes = audio_buffer.getvalue()
                audio_base64 = base64.b64encode(vibes).decode('utf-8')
                f.close()

            return audio_base64
            
        
    # --- Attach the synapse to the miner ----
    base_miner.axon.attach( MusicGen() )

    # --- Run Miner ----
    base_miner.run()

if __name__ == "__main__":
    bittensor.utils.version_checking()
    main( config() )





