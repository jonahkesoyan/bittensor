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

import time
import bittensor
from typing import List, Dict, Union, Tuple

bittensor.logging( bittensor.logging.config() )

class TextToSpeechSynapse( bittensor.TextToSpeechSynapse ):
    def priority( self, forward_call: "bittensor.SynapseCall" ) -> float:
        return 0.0

    def blacklist( self, forward_call: "bittensor.SynapseCall" ) -> Union[ Tuple[bool, str], bool ]:
        return False
    
    def forward( self, text: str ) -> bytes:
        return bytes( "these bytes represent the speech of text = {}".format( text ), 'utf-8')

# Create a mock wallet.
wallet = bittensor.wallet().create_if_non_existent()
axon = bittensor.axon( wallet = wallet, port = 9090, ip = "127.0.0.1" )
text_to_speech = bittensor.text_to_speech( axon = axon.info(), keypair = wallet.hotkey )
axon.attach( TextToSpeechSynapse( ) )

# Start the server and then exit after 50 seconds.
axon.start()
text = "lalala I am singing a song."
print( 'text =', text )
print( 'speech =', text_to_speech( text ).speech )
time.sleep(50)
axon.stop()