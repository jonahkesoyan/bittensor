import bittensor
import json
import asyncio

bittensor.logging(debug=True)
# Create a mock wallet.
wallet = bittensor.wallet(name='prompt', hotkey='default')
subtensor = bittensor.subtensor( chain_endpoint = 'wss://test.finney.opentensor.ai', network='finney' )
metagraph = subtensor.metagraph(21)

# Create a local endpoint receptor grpc connection.
local_endpoint = bittensor.endpoint(
    version=bittensor.__version_as_int__,
    uid=2,
    ip="127.0.0.1",
    ip_type=4,
    port=8091,
    hotkey=wallet.hotkey.ss58_address,
    coldkey=wallet.coldkeypub.ss58_address,
    modality=0,
)


def uid_selector():
    """
        Returns the uids of metagraph endpoint objs that are not equal to 0.0.0.0

        Returns:
            uids (list): The list of uids.
    """
    endpoints = []
    for endpoint in metagraph.endpoint_objs:
        if endpoint.ip != '0.0.0.0':
            print('ip |', endpoint.ip, '| uid |', endpoint.uid)
            endpoints.append(endpoint)
    return endpoints

# Create a text_prompting module and call it.
prompt = '''
You are Chattensor.
Chattensor is a research project by Opentensor Cortex.
Chattensor is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Chattensor is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Chattensor is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Chattensor is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.
Overall, Chattensor is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Chattensor is here to assist.
If asked about what Bittensor is, Chattensor will respond with the following:
   Bittensor is a peer to peer, decentralized network that facilitates a marketplace for machine intelligence.
   Validators utilize yuma consensus to rank and reward neural networks for their novel knowledge contributions.
   Servers deploy their machine intelligence to be ranked and rewarded by the network.
   This marketplace serves multi modalities of machine intelligence, including text, images, and audio through the finney subnetwork upgrade.
If asked about who the creators of Chattensor are, Chattensor will respond with the following:
    - Carro
    - Prism
Lists should not be long, like so:
• first item
• second item
• third item
They are the creators of Chattensor, but not Bittensor. That was founded by Jacob Steeves (Const) and Ala Shaabana (Shibshib). 
The current maintainers of Bittensor is the Opentensor Foundation. Carro and Prism work at Opentensor.'''

message = "who are you?"
timeout = 3
# for endpoint in uid_selector():
#     module = bittensor.text_prompting( endpoint = endpoint, wallet = wallet )
#     response = module.forward(
#         # messages = [json.dumps({ "role": "user", "content": "hello"})],
#         roles=["system", "user"],
#         messages = [prompt, "who are you?"],
#         timeout=1e6
#     )
#     print(response.response)


# async def run():
#     for endpoint in uid_selector():
#         module = bittensor.text_prompting( endpoint = endpoint, wallet = wallet )
#         response = await module.async_forward(
#             # messages = [json.dumps({ "role": "user", "content": "hello"})],
#             roles=["system", "user"],
#             messages = [prompt, "who are you?"],
#             timeout=12
#         )
#         print(response.response)

uids = [ endpoint.uid for endpoint in uid_selector() ]
dendrites = []
for uid, endpoint in enumerate( metagraph.endpoint_objs ):
    module = bittensor.text_prompting( endpoint = endpoint, wallet = wallet )
    dendrites.append( module )

if prompt is not None: 
    roles = ['system', 'user']
    messages = [ prompt, message ]
else:
    roles = ['user']
    messages = [ message ]

async def call_single_uid( uid: int ) -> str:
    print ('calling uid:', uid)
    module = bittensor.text_prompting( endpoint = metagraph.endpoint_objs[ uid ], wallet = wallet )
    response = await module.async_forward( 
        roles = roles, 
        messages = messages, 
        timeout = timeout 
    )
    print ('response:', response.response)
    return response.response

async def query():
    coroutines = [ call_single_uid( uid ) for uid in uids ]                
    all_responses = await asyncio.gather(*coroutines)
    print ('all responses:', all_responses)
    return all_responses


asyncio.run(query())

