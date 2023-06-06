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
import bittensor
from fastapi import APIRouter
from typing import List, Union, Callable
from abc import ABC, abstractmethod
from bittensor import Synapse, BaseSynapseCall

class TextToEmbeddingCall( BaseSynapseCall ):
    text: str
    embedding: List[List[float]]

class TextToEmbeddingSynapse( Synapse ):

    @property
    def name( self ) -> str: 
        return "text_to_embedding"

    def get_inputs_shape( self, call: BaseSynapseCall ) -> str: 
        return len( call.text )

    def get_outputs_shape( self, call: BaseSynapseCall ) -> str: 
        return len( call.embedding ) 

    def __init__( self, axon: "bittensor.axon" ):
        self.axon = axon
        self.router = APIRouter()
        self.router.add_api_route( "/TextToEmbedding/Forward/", self.forward_text_to_embedding, methods = [ "GET", "POST" ] )
        self.axon.fastapi_app.include_router( self.router )

    @abstractmethod
    def forward( self, text: str ) -> torch.Tensor: 
        ...

    def _preproccess_forward( self, call: TextToEmbeddingCall ) -> TextToEmbeddingCall: 
        embedding = forward( call.text )
        call.embedding = forward.tolist()
        return call

    def forward_text_to_embedding( self, call: TextToEmbeddingCall ) -> TextToEmbeddingCall:
        return self.apply( call = call, callable = self._preproccess_forward )
   