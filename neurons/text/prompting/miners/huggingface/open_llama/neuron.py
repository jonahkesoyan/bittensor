import torch
import bittensor
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

class CustomLlamaMiner(bittensor.HuggingFaceMiner):
    arg_prefix = "custom_llama"
    system_label = "\nSystem:"
    assistant_label = "\nAssistant:"
    user_label = "\nUser:"

    def load_tokenizer(self):
        return AutoTokenizer.from_pretrained(
            self.config.custom_llama.model_name, use_fast=True
        )

    def load_model(self):
        return AutoModelForCausalLM.from_pretrained(
            self.config.custom_llama.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )

    def forward(self, messages: List[Dict[str, str]]) -> str:
        history = self.process_history(messages)
        prompt = history + self.assistant_label

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(
            self.config.custom_llama.device
        )
        output = self.model.generate(
            input_ids,
            max_length=input_ids.shape[1] + self.config.custom_llama.max_new_tokens,
            temperature=0.7,
            do_sample=self.config.custom_llama.do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generation = self.tokenizer.decode(
            output[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        generation = generation.split("User:")[0].strip()

        bittensor.logging.debug("Prompt: " + str(prompt))
        bittensor.logging.debug("Message: " + str(messages))
        bittensor.logging.debug("Generation: " + str(generation).replace("<", "-"))
        return generation

if __name__ == "__main__":
    bittensor.utils.version_checking()
    CustomLlamaMiner().run()
