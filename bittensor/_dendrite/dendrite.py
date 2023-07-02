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

import uuid
import grpc
import time
import torch
import asyncio
import bittensor
import httpx 

from typing import Union, Callable, List, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class DendriteCall( ABC ):
    """ Base class for all dendrite calls."""

    is_forward: bool
    name: str

    def __init__(
            self,
            dendrite: 'bittensor.Dendrite',
            timeout: float = bittensor.__blocktime__
        ):
        self.dendrite = dendrite
        self.completed = False
        self.timeout = timeout
        self.start_time = time.time()
        self.elapsed_time = 0.0
        self.src_version = bittensor.__version_as_int__
        self.dest_hotkey = self.dendrite.axon_info.hotkey
        self.dest_version = self.dendrite.axon_info.version
        self.return_code: bittensor.proto.ReturnCode = bittensor.proto.ReturnCode.Success
        self.return_message: str = 'Success'

    def __repr__(self) -> str:
        return f"DendriteCall( {bittensor.utils.codes.code_to_string(self.return_code)}, to:{self.dest_hotkey[:4]} + ... + {self.dest_hotkey[-4:]}, msg:{self.return_message})"

    def __str__(self) -> str:
        return self.__repr__()

    @abstractmethod
    def get_inputs_shape(self) -> torch.Size: ...

    @abstractmethod
    def get_outputs_shape(self) -> torch.Size: ...

    @abstractmethod
    def get_fastapi_request_payload(self) -> object: ...

    @abstractmethod
    def apply_fast_api_response( self, response: object ): ...

    def end(self):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.completed = True

    @property
    def did_timeout( self ) -> bool: return self.return_code == bittensor.proto.ReturnCode.Timeout
    @property
    def is_success( self ) -> bool: return self.return_code == bittensor.proto.ReturnCode.Success
    @property
    def did_fail( self ) -> bool: return not self.is_success

    def log_outbound(self):
        bittensor.logging.rpc_log(
            axon = False,
            forward = self.is_forward,
            is_response = False,
            code = self.return_code,
            call_time = 0,
            pubkey = self.dest_hotkey,
            uid = self.dendrite.uid,
            inputs = self.get_inputs_shape(),
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name,
        )

    def log_inbound(self):
        bittensor.logging.rpc_log(
            axon = False,
            forward = self.is_forward,
            is_response = True,
            code = self.return_code,
            call_time = self.elapsed,
            pubkey = self.dest_hotkey,
            uid = self.dendrite.uid,
            inputs = self.get_inputs_shape(),
            outputs = self.get_outputs_shape(),
            message = self.return_message,
            synapse = self.name
        )

class Dendrite( ABC, torch.nn.Module ):
    def __init__(
            self,
            keypair: Union[ 'bittensor.Wallet', 'bittensor.Keypair'],
            axon: Union[ 'bittensor.axon_info', 'bittensor.axon' ],
            uid : int = 0,
            ip: str = None,
            grpc_options: List[Tuple[str,object]] =
                    [('grpc.max_send_message_length', -1),
                     ('grpc.max_receive_message_length', -1),
                     ('grpc.keepalive_time_ms', 100000) ]
        ):
        """ Dendrite abstract class
            Args:
                keypair (:obj:`Union[ 'bittensor.Wallet', 'bittensor.Keypair']`, `required`):
                    bittensor keypair used for signing messages.
                axon (:obj:Union[`bittensor.axon_info`, 'bittensor.axon'], `required`):
                    bittensor axon object or its info used to create the connection.
                grpc_options (:obj:`List[Tuple[str,object]]`, `optional`):
                    grpc options to pass through to channel.
        """
        super(Dendrite, self).__init__()
        self.uuid = str(uuid.uuid1())
        self.uid = uid
        self.ip = ip
        self.keypair = keypair.hotkey if isinstance( keypair, bittensor.Wallet ) else keypair
        self.axon_info = axon.info() if isinstance( axon, bittensor.axon ) else axon
        self.loop = asyncio.get_event_loop()

    async def apply( self, dendrite_call: 'DendriteCall' ) -> DendriteCall:
        """ Applies a dendrite call to the endpoint.
            Args:
                dendrite_call (:obj:`DendriteCall`, `required`):
                    Dendrite call to apply.
            Returns:
                DendriteCall: Dendrite call with response.
        """
        bittensor.logging.trace('Dendrite.apply()')
        try:
            dendrite_call.log_outbound()
            # Prepare headers including signature
            headers = {
                'rpc-auth-header': 'Bittensor',
                'bittensor-signature': self.sign(),
                'bittensor-version': str(bittensor.__version_as_int__),
            }
            # Prepare payload from dendrite_call._get_request_proto() here
            data = dendrite_call.get_fastapi_request_payload()

            # Form the request url
            url = f"http://{self.axon_info.ip}:{self.axon_info.external_fast_api_port}{dendrite_call.route}"

            # Send the request
            bittensor.logging.trace( 'Dendrite.apply() awaiting response from: {}'.format( self.axon_info.hotkey ) )
            async with httpx.AsyncClient() as client:
                response = await client.post(url, headers=headers, json=data)
            bittensor.logging.trace( 'Dendrite.apply() received response from: {}'.format( self.axon_info.hotkey ) )

            # Handle response and errors
            if response.status_code == 200:
                # Process the response if successful
                response_proto = response.json()
                dendrite_call.apply_fast_api_response( response_proto )
            else:
                # Raise an error if not successful
                raise Exception(response.text)
            bittensor.logging.trace( 'Dendrite.apply() received response from: {}'.format( self.axon_info.hotkey ) )

        # Other uncaught exceptions
        except Exception as e:
            dendrite_call.return_code = response.status_code
            dendrite_call.return_message = str(e)
            bittensor.logging.error( 'Dendrite.apply() received error from: {}'.format( self.axon_info.hotkey ) )
            bittensor.logging.error( e )

        finally:
            dendrite_call.end()
            dendrite_call.log_inbound()
            dendrite_call.elapsed_time = time.time() - dendrite_call.start_time
            return dendrite_call

    def __exit__ ( self ):
        self.__del__()

    def close ( self ):
        self.__exit__()

    def __del__ ( self ):
        pass

    def nonce ( self ):
        return time.monotonic_ns()

    def sign(self) -> str:
        """ Creates a signature for the dendrite and returns it as a string."""
        nonce = f"{self.nonce()}"
        sender_hotkey = self.keypair.ss58_address
        receiver_hotkey = self.axon_info.hotkey
        message = f"{nonce}.{sender_hotkey}.{receiver_hotkey}.{self.uuid}"
        signature = f"0x{self.keypair.sign(message).hex()}"
        return ".".join([nonce, sender_hotkey, signature, self.uuid])

    def state ( self ):
        """ Returns the state of the dendrite channel."""
        try:
            return self.state_dict[self.channel._channel.check_connectivity_state(True)]
        except ValueError:
            return "Channel closed"







