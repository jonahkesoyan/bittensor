
from typing import Callable, Tuple, Union
from abc import ABC, abstractmethod

class BaseSynapseCall( BaseModel ):
    hotkey: str
    timeout: float
    signature: bytes 
    return_code: int = 0
    return_message: str = 'NotFilled'

    def verify(self) -> bool:
        """ Verify the signature of this call.
        """
        self.keypair = bittensor.Keypair( ss58_address = self.hotkey )
        return self.keypair.verify( data = self.json(), signature = self.signature )

class Synapse( ABC ):

    @abstractmethod
    @property
    def name( self ) -> str: ...

    @abstractmethod
    def blacklist( self, call: BaseSynapseCall ) -> Union[ Tuple[bool, str], bool ]: ...

    @abstractmethod
    def priority( self, call: BaseSynapseCall ) -> float: ...

    @abstractmethod
    def get_inputs_shape( self, call: BaseSynapseCall ) -> str: ...

    @abstractmethod
    def get_outputs_shape( self, call: BaseSynapseCall ) -> str: ...

    def apply( self, call: Call, apply_callable: Callable ) -> object:
        bittensor.logging.trace( 'Synapse: {} received call: {}'.format( self.name(), call ) )
        start_time = time.time()
        try:
            bittensor.logging.rpc_log(
                axon = True,
                forward = True,
                is_response = False,
                code = call.return_code,
                call_time = 0,
                pubkey = call.hotkey
                uid = None,
                inputs = self.get_inputs_shape( call ),
                outputs = None,
                message = call.return_message,
                synapse = self.name()
            )

            # 1. Check verification.
            if not call.verify():
                call.return_code = 23 # Unverified.
                call.return_message = 'Signature verification failed.'
                bittensor.logging.info( 'Synapse: {} blacklisted call: {} reason: {}'.format( self.name(), call, reason ) )

            # 2. Check blacklist.
            blacklist, reason = self.blacklist( call )
            elif blacklist:
                call.return_code = 25 # Blacklisted
                call.return_message = reason
                bittensor.logging.info( 'Synapse: {} blacklisted call: {} reason: {}'.format( self.name(), call, reason) )

            # 3. If all passes, make call.
            else:
                # Queue the forward call with priority.
                priority = self.priority( call )
                future = self.axon.priority_threadpool.submit(
                    apply_callable,
                    priority = priority,
                )
                bittensor.logging.trace( 'Synapse: {} loaded future: {}'.format( self.name(), future ) )
                future.result( timeout = call.timeout )
                bittensor.logging.trace( 'Synapse: {} completed call: {}'.format( self.name(), call ) )

        # Catch timeouts
        except asyncio.TimeoutError:
            bittensor.logging.trace( 'Synapse: {} timeout: {}'.format( self.name(), call.timeout ) )
            call.return_code = 2 # Timeout
            call.return_message = 'GRPC request timeout after: {}s'.format( call.timeout)

        # Catch unknown exceptions.
        except Exception as e:
            bittensor.logging.trace( 'Synapse: {} unknown error: {}'.format( self.name(), str(e) ) )
            call.return_code = 22 # Unknown
            call.return_message = str( e )

        # Finally return the call.
        finally:
            bittensor.logging.trace( 'Synapse: {} finalize call {}'.format( self.name(), call ) )
            bittensor.logging.rpc_log(
                axon = True,
                forward = True,
                is_response = True,
                code = call.return_code,
                call_time = time.time() - start_time,
                pubkey = call.hotkey,
                uid = None,
                inputs = None,
                outputs = self.get_outputs_shape( call ),
                message = call.return_message,
                synapse = self.name(),
            )
            return call