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

import bittensor
from fastapi import APIRouter
from typing import List, Union, Callable, Dict
from abc import ABC, abstractmethod
from bittensor import Synapse, BaseSynapseCall

class ForwardTextToTextCall( BaseSynapseCall ):
    roles: List[ str ]
    messages: List[ str ]
    completion: str

class BackwardTextToTextCall( BaseSynapseCall ):
    roles: List[ str ]
    messages: List[ str ]
    completion: str
    rewards: List[ float ]

class TextToEmbeddingSynapse( Synapse ):

    @property
    def name( self ) -> str: 
        return "text_to_text"

    def get_inputs_shape( self, call: BaseSynapseCall ) -> str: 
        return len( call.messages )

    def get_outputs_shape( self, call: BaseSynapseCall ) -> str: 
        return len( call.completion ) 

    def __init__( self, axon: "bittensor.axon" ):
        self.axon = axon
        self.router = APIRouter()
        self.router.add_api_route( "/TextToCompletion/Forward/", self.forward_text_to_text, methods = ["GET", "POST"])
        self.router.add_api_route( "/TextToCompletion/Backward", self.backward_text_to_text, methods = ["GET", "POST"])
        self.axon.fastapi_app.include_router( self.router )

    @abstractmethod
    def forward( self, messages: List[Dict[str, str]] ) -> str: ...

    @abstractmethod
    def backward( self, roles: List[ str ], messages: List[ str ], completion: str, rewards: List[ float ] ) -> str: ...

    def _preproccess_forward( self, call: ForwardTextToTextCall ): 
        call.completion = self.forward( call.messages )

    def _preproccess_backward( self, call: BackwardTextToTextCall ): 
        self.backward( message = call.messages, completion = call.completion, rewards = call.rewards )
        
    def forward_text_to_text( self, call: ForwardTextToTextCall ) -> ForwardTextToTextCall:
        return self.apply( call = call, callable = self._preproccess_forward )

    def backward_text_to_text( self, call: BackwardTextToTextCall ) -> BackwardTextToTextCall:
        return self.apply( call = call, callable = self._preproccess_forward )