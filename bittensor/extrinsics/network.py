# The MIT License (MIT)
# Copyright © 2021 Yuma Rao
# Copyright © 2023 Opentensor Foundation

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
import json
import bittensor
import bittensor.utils.networking as net
from dataclasses import asdict
from rich.prompt import Confirm


def register_subnetwork_extrinsic(
    subtensor: "bittensor.subtensor",
    wallet: "bittensor.wallet",
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = True,
    prompt: bool = False,
) -> bool:
    r"""Removes the wallet from chain for senate voting.
    Args:
        wallet (bittensor.wallet):
            bittensor wallet object.
        wait_for_inclusion (bool):
            If set, waits for the extrinsic to enter a block before returning true,
            or returns false if the extrinsic fails to enter the block within the timeout.
        wait_for_finalization (bool):
            If set, waits for the extrinsic to be finalized on the chain before returning true,
            or returns false if the extrinsic fails to be finalized within the timeout.
        prompt (bool):
            If true, the call waits for confirmation from the user before proceeding.
    Returns:
        success (bool):
            flag is true if extrinsic was finalized or included in the block.
            If we did not wait for finalization / inclusion, the response is true.
    """
    wallet.coldkey  # unlock coldkey

    if prompt:
        # Prompt user for confirmation.
        if not Confirm.ask(f"Register subnet?"):
            return False

    with bittensor.__console__.status(":satellite: Registering subnet..."):
        with subtensor.substrate as substrate:
            # create extrinsic call
            call = substrate.compose_call(
                call_module="SubtensorModule",
                call_function="register_network",
                call_params={"immunity_period": 0, "reg_allowed": True},
            )
            extrinsic = substrate.create_signed_extrinsic(
                call=call, keypair=wallet.coldkey
            )
            response = substrate.submit_extrinsic(
                extrinsic,
                wait_for_inclusion=wait_for_inclusion,
                wait_for_finalization=wait_for_finalization,
            )

            # We only wait here if we expect finalization.
            if not wait_for_finalization and not wait_for_inclusion:
                return True

            # process if registration successful
            response.process_events()
            if not response.is_success:
                bittensor.__console__.print(
                    ":cross_mark: [red]Failed[/red]: error:{}".format(
                        response.error_message
                    )
                )
                time.sleep(0.5)

            # Successful registration, final check for membership
            else:
                bittensor.__console__.print(
                    ":white_heavy_check_mark: [green]Registered subnetwork[/green]"
                )
                return True