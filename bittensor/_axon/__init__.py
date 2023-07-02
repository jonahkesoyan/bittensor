""" Create and init Axon, whcih services Forward and Backward requests from other neurons.
"""
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2022 Opentensor Foundation

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
import os
import copy
import torch
import argparse
import bittensor

# os.environ["UVICORN_LOOP"] = "asyncio"

from dataclasses import dataclass
from substrateinterface import Keypair
import bittensor.utils.networking as net
from typing import Optional, Tuple, Dict, Optional, List

import time
import threading
import contextlib
import uvicorn
from fastapi import FastAPI, APIRouter, Request, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint


class FastAPIThreadedServer(uvicorn.Server):
    """ FastAPI server that runs in a thread."""

    should_exit: bool = False
    is_running: bool = False

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

    def _wrapper_run(self):
        with self.run_in_thread():
            while not self.should_exit:
                time.sleep(1e-3)

    def start(self):
        if not self.is_running:
            self.should_exit = False
            thread = threading.Thread(target=self._wrapper_run, daemon=True)
            thread.start()
            self.is_running = True

    def stop(self):
        if self.is_running:
            self.should_exit = True


class axon:
    """ Axon object for serving synapse receptors. """

    def info(self) -> "axon_info":
        """Returns the axon info object associate with this axon."""
        return axon_info(
            version=bittensor.__version_as_int__,
            ip=self.external_ip,
            ip_type=4,
            port=self.external_fast_api_port,
            hotkey=self.wallet.hotkey.ss58_address,
            coldkey=self.wallet.coldkeypub.ss58_address,
            protocol=4,
            placeholder1=0,
            placeholder2=0,
        )

    def fast_api_info(self) -> dict:
        fastinfo = self.info().__dict__
        fastinfo.update({"fast_api_port": self.config.axon.fast_api_port})
        fastinfo.update(
            {"external_fast_api_port": self.config.axon.external_fast_api_port}
        )
        return fastinfo

    def __init__(
        self,
        wallet: "bittensor.Wallet",
        metagraph: Optional["bittensor.Metagraph"] = None,
        config: Optional["bittensor.config"] = None,
        port: Optional[int] = None,
        ip: Optional[str] = None,
        external_ip: Optional[str] = None,
        external_port: Optional[int] = None,
        max_workers: Optional[int] = None,
        maximum_concurrent_rpcs: Optional[int] = None,
        disable_fast_api: Optional[bool] = None,
        fast_api_port: Optional[int] = None,
        external_fast_api_port: Optional[int] = None,
    ) -> "bittensor.Axon":
        r"""Creates a new bittensor.Axon object from passed arguments.
        Args:
            config (:obj:`Optional[bittensor.Config]`, `optional`):
                bittensor.axon.config()
            wallet (:obj:`Optional[bittensor.Wallet]`, `optional`):
                bittensor wallet with hotkey and coldkeypub.
            port (:type:`Optional[int]`, `optional`):
                Binding port.
            ip (:type:`Optional[str]`, `optional`):
                Binding ip.
            external_ip (:type:`Optional[str]`, `optional`):
                The external ip of the server to broadcast to the network.
            external_port (:type:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network.
            max_workers (:type:`Optional[int]`, `optional`):
                Used to create the threadpool if not passed, specifies the number of active threads servicing requests.
            maximum_concurrent_rpcs (:type:`Optional[int]`, `optional`):
                Maximum allowed concurrently processed RPCs.
            disable_fast_api (:obj:`Optional[bool]`, `optional`):
                Specifies whether to disable the FastAPI server. If set to `True`, the Axon will not start the FastAPI server.
            fast_api_port (:obj:`Optional[int]`, `optional`)
                The port number on which the FastAPI server should bind. If specified, the FastAPI server will bind to this port.
            external_fast_api_port (:obj:`Optional[int]`, `optional`):
                The external port of the server to broadcast to the network for FastAPI requests.
        """
        self.metagraph = metagraph
        self.wallet = wallet

        # Build and check config.
        if config is None:
            config = axon.config()
        config = copy.deepcopy(config)
        config.axon.port = port if port is not None else config.axon.port
        config.axon.ip = ip if ip is not None else config.axon.ip
        config.axon.external_ip = (
            external_ip if external_ip is not None else config.axon.external_ip
        )
        config.axon.external_port = (
            external_port if external_port is not None else config.axon.external_port
        )
        config.axon.disable_fast_api = disable_fast_api
        config.axon.fast_api_port = fast_api_port or config.axon.fast_api_port
        config.axon.external_fast_api_port = (
            external_fast_api_port or config.axon.external_fast_api_port
        )
        config.axon.max_workers = (
            max_workers if max_workers is not None else config.axon.max_workers
        )
        config.axon.maximum_concurrent_rpcs = (
            maximum_concurrent_rpcs
            if maximum_concurrent_rpcs is not None
            else config.axon.maximum_concurrent_rpcs
        )
        axon.check_config(config)
        self.config = config

        # Build axon objects.
        self.ip = self.config.axon.ip
        self.port = self.config.axon.port
        self.external_ip = (
            self.config.axon.external_ip
            if self.config.axon.external_ip != None
            else bittensor.utils.networking.get_external_ip()
        )
        self.external_fast_api_port = (
            self.config.axon.external_fast_api_port
            if self.config.axon.external_fast_api_port != None
            else self.config.axon.fast_api_port
        )
        self.full_address = str(self.config.axon.ip) + ":" + str(self.config.axon.port)
        self.started = False

        # Build interceptors.
        self.receiver_hotkey = self.wallet.hotkey.ss58_address

        # Instantiate FastAPI
        self.fastapi_app = FastAPI()
        # Add interceptor for fastAPI queries.
        # NOTE: All requests will depend on the fast_interceptor.
        self.fastapi_app.add_middleware(
            FastAuthInterceptor, receiver_hotkey=self.receiver_hotkey
        )
        self.fast_config = uvicorn.Config(
            self.fastapi_app,
            host="0.0.0.0",
            port=self.config.axon.fast_api_port,
            log_level="info",
        )
        self.server = FastAPIThreadedServer(config=self.fast_config)
        self.router = APIRouter()
        self.router.add_api_route("/", self.fast_api_info, methods=["GET", "POST"])
        self.fastapi_app.include_router(self.router)

        # Build priority thread pool
        self.priority_threadpool = bittensor.prioritythreadpool(config=self.config.axon)

    @classmethod
    def config(cls) -> "bittensor.Config":
        """Get config from the argument parser
        Return: bittensor.config object
        """
        parser = argparse.ArgumentParser()
        axon.add_args(parser)
        return bittensor.config(parser)

    @classmethod
    def help(cls):
        """Print help to stdout"""
        parser = argparse.ArgumentParser()
        cls.add_args(parser)
        print(cls.__new__.__doc__)
        parser.print_help()

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser, prefix: str = None):
        """Accept specific arguments from parser"""
        prefix_str = "" if prefix is None else prefix + "."
        if prefix is not None:
            if not hasattr(bittensor.defaults, prefix):
                setattr(bittensor.defaults, prefix, bittensor.Config())
            getattr(bittensor.defaults, prefix).axon = bittensor.defaults.axon

        bittensor.prioritythreadpool.add_args(parser, prefix=prefix_str + "axon")
        try:
            parser.add_argument(
                "--" + prefix_str + "axon.port",
                type=int,
                help="""The local port this axon endpoint is bound to. i.e. 8091""",
                default=bittensor.defaults.axon.port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.ip",
                type=str,
                help="""The local ip this axon binds to. ie. [::]""",
                default=bittensor.defaults.axon.ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_port",
                type=int,
                required=False,
                help="""The public port this axon broadcasts to the network. i.e. 8091""",
                default=bittensor.defaults.axon.external_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_ip",
                type=str,
                required=False,
                help="""The external ip this axon broadcasts to the network to. ie. [::]""",
                default=bittensor.defaults.axon.external_ip,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.fast_api_port",
                type=int,
                help="""The local port this axon fast api endpoint is bound to. i.e. 8092""",
                default=bittensor.defaults.axon.fast_api_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.external_fast_api_port",
                type=int,
                required=False,
                help="""The public fast api port this axon broadcasts to the network. i.e. 8092""",
                default=bittensor.defaults.axon.external_fast_api_port,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.max_workers",
                type=int,
                help="""The maximum number connection handler threads working simultaneously on this endpoint.
                        The grpc server distributes new worker threads to service requests up to this number.""",
                default=bittensor.defaults.axon.max_workers,
            )
            parser.add_argument(
                "--" + prefix_str + "axon.maximum_concurrent_rpcs",
                type=int,
                help="""Maximum number of allowed active connections""",
                default=bittensor.defaults.axon.maximum_concurrent_rpcs,
            )
        except argparse.ArgumentError:
            # re-parsing arguments.
            pass

    @classmethod
    def add_defaults(cls, defaults):
        """Adds parser defaults to object from enviroment variables."""
        defaults.axon = bittensor.Config()
        defaults.axon.port = (
            os.getenv("BT_AXON_PORT") if os.getenv("BT_AXON_PORT") is not None else 8091
        )
        defaults.axon.ip = (
            os.getenv("BT_AXON_IP") if os.getenv("BT_AXON_IP") is not None else "[::]"
        )
        defaults.axon.fast_api_port = os.getenv("BT_AXON_FAST_API_PORT") or 8092
        defaults.axon.external_port = (
            os.getenv("BT_AXON_EXTERNAL_PORT")
            if os.getenv("BT_AXON_EXTERNAL_PORT") is not None
            else None
        )
        defaults.axon.external_fast_api_port = (
            os.getenv("BT_AXON_EXTERNAL_FAST_API_PORT") or None
        )
        defaults.axon.external_ip = (
            os.getenv("BT_AXON_EXTERNAL_IP")
            if os.getenv("BT_AXON_EXTERNAL_IP") is not None
            else None
        )
        defaults.axon.max_workers = (
            os.getenv("BT_AXON_MAX_WORERS")
            if os.getenv("BT_AXON_MAX_WORERS") is not None
            else 10
        )
        defaults.axon.maximum_concurrent_rpcs = (
            os.getenv("BT_AXON_MAXIMUM_CONCURRENT_RPCS")
            if os.getenv("BT_AXON_MAXIMUM_CONCURRENT_RPCS") is not None
            else 400
        )
        defaults.axon.disable_fast_api = os.getenv("BT_AXON_DISABLE_FAST_API") or False

    @classmethod
    def check_config(cls, config: "bittensor.Config"):
        """Check config for axon port and wallet"""
        assert (
            config.axon.port > 1024 and config.axon.port < 65535
        ), "port must be in range [1024, 65535]"
        assert config.axon.external_port is None or (
            config.axon.external_port > 1024 and config.axon.external_port < 65535
        ), "external port must be in range [1024, 65535]"

    def __str__(self) -> str:
        return "Axon({}, {}, {}, {})".format(
            self.ip,
            self.port,
            self.wallet.hotkey.ss58_address,
            "started" if self.started else "stopped",
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __del__(self):
        r"""Called when this axon is deleted, ensures background threads shut down properly."""
        self.stop()

    def start(self) -> "bittensor.axon":
        r"""Starts the standalone axon fastAPI server thread."""
        # if self.server is not None:
        #     self.server.stop()
        self.server.start()
        self.started = True
        return self

    def stop(self) -> "bittensor.axon":
        r"""Stop the axon fastAPI server."""
        if hasattr(self, "server") and self.server is not None:
            self.server.stop()
        self.started = False


class FastAuthInterceptor(BaseHTTPMiddleware):
    def __init__(self, app, receiver_hotkey):
        super().__init__(app)
        self.nonces = {}
        self.receiver_hotkey = receiver_hotkey

    def parse_signature_v2(self, signature: str) -> Optional[Tuple[int, str, str, str]]:
        r"""Attempts to parse a signature using the v2 format"""
        parts = signature.split(".")
        if len(parts) != 4:
            return None
        try:
            nonce = int(parts[0])
        except ValueError:
            return None
        sender_hotkey = parts[1]
        signature = parts[2]
        receptor_uuid = parts[3]
        return (nonce, sender_hotkey, signature, receptor_uuid)

    def parse_signature(self, metadata: Dict[str, str]) -> Tuple[int, str, str, str]:
        r"""Attempts to parse a signature from the metadata"""
        signature = metadata.get("bittensor-signature")
        version = metadata.get("bittensor-version")
        if signature is None:
            raise HTTPException(status_code=400, detail="Request signature missing")
        if int(version) < 510:
            raise HTTPException(status_code=400, detail="Incorrect Version")
        parts = self.parse_signature_v2(signature)
        if parts is not None:
            return parts
        raise HTTPException(status_code=400, detail="Unknown signature format")

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint):
        metadata = dict(request.headers)
        try:
            (nonce, sender_hotkey, signature, receptor_uuid) = self.parse_signature(
                metadata
            )
        except HTTPException as e:
            # Error parsing signature
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
        except Exception as e:
            # Some other exception occurred
            return JSONResponse({"detail": str(e)}, status_code=400)

        try:
            self.check_signature(int(nonce), sender_hotkey, signature, receptor_uuid)
        except HTTPException as e:
            # Error checking signature
            return JSONResponse({"detail": e.detail}, status_code=e.status_code)
        except Exception as e:
            # Some other exception occurred
            return JSONResponse({"detail": str(e)}, status_code=400)

        response = await call_next(request)
        return response

    def check_signature(
        self, nonce: int, sender_hotkey: str, signature: str, receptor_uuid: str
    ):
        keypair = Keypair(ss58_address=sender_hotkey)
        message = f"{nonce}.{sender_hotkey}.{self.receiver_hotkey}.{receptor_uuid}"
        endpoint_key = f"{sender_hotkey}:{receptor_uuid}"

        if endpoint_key in self.nonces.keys():
            previous_nonce = self.nonces[endpoint_key]
            if nonce <= previous_nonce:
                raise HTTPException(status_code=403, detail="Nonce is too small")

        if not keypair.verify(message, signature):
            raise HTTPException(status_code=403, detail="Signature mismatch")

        self.nonces[endpoint_key] = nonce


METADATA_BUFFER_SIZE = 250


@dataclass
class axon_info:

    version: int
    ip: str
    port: int
    ip_type: int
    hotkey: str
    coldkey: str
    protocol: int = 4
    placeholder1: int = 0
    placeholder2: int = 0

    @property
    def is_serving(self) -> bool:
        """ True if the endpoint is serving. """
        if self.ip == "0.0.0.0":
            return False
        else:
            return True

    def ip_str(self) -> str:
        """ Return the whole ip as string """
        return net.ip__str__(self.ip_type, self.ip, self.port)

    def __eq__(self, other: "axon_info"):
        if other == None:
            return False
        if (
            self.version == other.version
            and self.ip == other.ip
            and self.port == other.port
            and self.ip_type == other.ip_type
            and self.coldkey == other.coldkey
            and self.hotkey == other.hotkey
        ):
            return True
        else:
            return False

    def __str__(self):
        return "axon_info( {}, {}, {}, {} )".format(
            str(self.ip_str()), str(self.hotkey), str(self.coldkey), self.version
        )

    def __repr__(self):
        return self.__str__()

    @classmethod
    def from_neuron_info(cls, neuron_info: dict) -> "axon_info":
        """ Converts a dictionary to an axon_info object. """
        return cls(
            version=neuron_info["axon_info"]["version"],
            ip=bittensor.utils.networking.int_to_ip(
                int(neuron_info["axon_info"]["ip"])
            ),
            port=neuron_info["axon_info"]["port"],
            ip_type=neuron_info["axon_info"]["ip_type"],
            hotkey=neuron_info["hotkey"],
            coldkey=neuron_info["coldkey"],
        )

    def to_parameter_dict(self) -> "torch.nn.ParameterDict":
        r""" Returns a torch tensor of the subnet info.
        """
        return torch.nn.ParameterDict(self.__dict__)

    @classmethod
    def from_parameter_dict(
        cls, parameter_dict: "torch.nn.ParameterDict"
    ) -> "axon_info":
        r""" Returns an axon_info object from a torch parameter_dict.
        """
        return cls(**dict(parameter_dict))
