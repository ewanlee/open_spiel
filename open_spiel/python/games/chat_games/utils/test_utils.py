# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for running tests."""

import dataclasses
import enum

from typing import List

import numpy as np

from open_spiel.python.games.chat_games.envs.comm_substrates import emails
from open_spiel.python.games.chat_games.envs.comm_substrates.emails import CHAR_MSG

from transformers import (
    AutoTokenizer,
    AutoModel,
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
)
import transformers
import torch
from torch.nn.functional import softmax, log_softmax
import random

# PATH = "/home/ewanlee/Models/llama-2-7b-chat-hf/models/"
# PATH = "/home/ewanlee/Models/llama-2-7b-hf/models/"
PATH = "/home/ewanlee/Models/llama-3-8b-instruct/models/"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate_predefined_outputs(model, tokenizer, input_prompt, predefined_outputs, seed):
    set_seed(seed)
    input_tokens = tokenizer(input_prompt, return_tensors="pt").input_ids.to(model.device)
    log_probabilities = []

    for output in predefined_outputs:
        output_tokens = tokenizer(output, return_tensors="pt").input_ids.to(model.device)
        concatenated_tokens = torch.cat((input_tokens, output_tokens[:, 1:]), dim=1) # Remove the first token (BOS) from the output_tokens
        logits = model(concatenated_tokens).logits
        output_logits = logits[:, -output_tokens.shape[1]:, :].squeeze(0) # Extract logits for output tokens only

        # Calculate the sum of log probabilities for the output tokens
        log_token_probs = log_softmax(output_logits, dim=-1)
        # output_token_ids = output_tokens.squeeze(0)[1:] # Remove the first token (BOS)
        output_token_ids = output_tokens.squeeze(0)
        log_prob_sum = log_token_probs[range(len(output_token_ids)), output_token_ids].sum().item()

        log_probabilities.append(log_prob_sum)

    return log_probabilities



class TestLLM(enum.Enum):
    MOCK = 0
    LLAMA2CHAT = 1


@dataclasses.dataclass(frozen=True)
class Llama2ChatScore:
    logprob: float


class Llama2Response:
    """Llama2 LLM response."""

    def __init__(self, model, tokenizer, length: int, seed: int, 
                 system_message: str, user_message: str):
        self.model = model
        self.tokenizer = tokenizer
        self.length = length
        self.seed = seed
        self.system_message = system_message
        self.user_message = user_message
        self.conversation_history = []
        self.template = '{system_prompt}{user_message}'
        self.text = self.generate_response()

    def generate_response(self):
        prompt = self.template.format(
            system_prompt=self.system_message, 
            user_message=self.user_message)
        input_ids = self.tokenizer.encode(
            prompt, 
            return_tensors="pt", 
            add_special_tokens=False)
        start_idx = input_ids.shape[-1]

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.length,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.6,
            "repetition_penalty": 1.2,
            "num_beams": 1,
            # "output_scores": True,
            # "return_dict_in_generate": True,
            # "output_logits": True,
        }
        set_seed(self.seed)
        generate_ids = self.model.generate(**generate_kwargs)
        assistant_response = self.tokenizer.decode(generate_ids[0][start_idx:], skip_special_tokens=True)
        # self.conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response


class Llama2ChatResponse:
    """Llama2Chat LLM response."""

    def __init__(self, model, tokenizer, length: int, seed: int, 
                 system_message: str, user_message: str):
        self.model = model
        self.tokenizer = tokenizer
        self.length = length
        self.seed = seed
        self.system_message = system_message
        self.user_message = user_message
        self.conversation_history = []
        self.text = self.generate_response()

    def generate_response(self):
        system_prompt = self.system_message
        self.conversation_history.append({"role": "system", "content": system_prompt})

        self.conversation_history.append({"role": "user", "content": self.user_message})
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history, 
            return_tensors="pt", 
            add_generation_prompt=True, # for llama 3
            add_special_tokens=False)
        start_idx = input_ids.shape[-1]

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        # for llama 3
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": self.length,
            "do_sample": True,
            # "top_k": 50, # for llama 2
            "top_p": 0.9,
            "temperature": 0.6,
            # "repetition_penalty": 1.2, # for llama 2
            # "num_beams": 1, # for llama 2
            # "output_scores": True,
            # "return_dict_in_generate": True,
            # "output_logits": True,
            "eos_token_id": terminators, # for llama 3
            "pad_token_id": self.tokenizer.eos_token_id, # for llama 3
        }
        set_seed(self.seed)
        generate_ids = self.model.generate(**generate_kwargs)
        assistant_response = self.tokenizer.decode(generate_ids[0][start_idx:], skip_special_tokens=True)
        # self.conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response


class Llama2ChatTokenizer:
    """Llama2Chat Tokenizer."""

    def __init__(self, path=PATH):
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.use_default_system_prompt = True

    def to_int(self, text: str) -> np.ndarray:
        return self.tokenizer.encode(text, return_tensors="np", add_special_tokens=False)[0]
        # return np.zeros(len(text), dtype=np.int32)


class Llama2ChatVectorizer:
    """Llama2Chat Vectorizer."""

    def __init__(self):
        self.tokenizer = Llama2ChatTokenizer()

    def vectorize(self, text: str, obs_size: int) -> np.ndarray:
        observation = self.tokenizer.to_int(text)[:obs_size]
        num_pad = max(0, obs_size - observation.size)
        observation = np.pad(observation, (0, num_pad))
        return observation


class Llama2ChatModel:
    """Llama2Chat model."""

    def __init__(self, path: str):
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            # torch_dtype=torch.float16, # for llama 2
            torch_dtype=torch.bfloat16, # for llama 3
            # load_in_8bit=True,
            trust_remote_code=True,
            # use_flash_attention_2=True,
        )
        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        # self.tokenizer.pad_token = self.tokenizer.eos_token # for llama 2
        self.tokenizer.use_default_system_prompt = True
        # self.conversation_history = []

        # system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        # self.conversation_history.append({"role": "system", "content": system_prompt})
    

class Llama2ChatClient:
    """Llama2Chat LLM client."""

    def __init__(self):
        # for cycling through mock response options
        self._idxs = {"names": 0, "tones": 0, "examples": 0}

    def sample(self, model, tokenizer, length: int, seed: int, prompt: str) -> Llama2ChatResponse:
        """Returns string responses according to fixed prompt styles."""
        prompt_lower = prompt.lower()
        info_keys = ["names", "tones", "fruit_endowment", "fruit_valuations"]
        # Llama2ChatResponse = Llama2Response
        if "read the following summary of a dialgoue between two parties attempting to reach a trade agreement" in prompt_lower:
            return Llama2ChatResponse(model, tokenizer, length, seed, '', prompt)
        elif ("summary" in prompt_lower) and ("extract out the final value" not in prompt_lower):
            start_idx = prompt_lower.rindex('please summarize the following')
            system_message = prompt[:start_idx]
            user_message = prompt[start_idx:]
            return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        elif "report how much" in prompt_lower:
            start_idx = prompt_lower.rindex('dialogue:')
            system_message = prompt[:start_idx]
            user_message = prompt[start_idx:]
            return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        elif "calculate the values" in prompt_lower:
            start_index = prompt.index('Now')
            system_message = prompt[:start_index]
            user_message = prompt[start_index:]
            return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        elif "extract out the final value" in prompt_lower:
            return Llama2ChatResponse(model, tokenizer, length, seed, "", prompt)
        elif "now" in prompt_lower: # generate chat reply
            now_index = prompt.index('Now')
            substring_after_now = prompt[now_index:]
            char_msg_index = substring_after_now.index(CHAR_MSG)
            system_message = prompt[:now_index + char_msg_index]
            user_message = substring_after_now[char_msg_index:]
            return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        # for key in info_keys:
        #     if key in prompt_lower:
        #         start_idx = prompt_lower.index('input:')
        #         system_message = prompt[:start_idx]
        #         user_message = prompt[start_idx:]
        #         return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        elif "continue the list" in prompt_lower:
            start_idx = prompt_lower.index('input:')
            system_message = prompt[:start_idx]
            user_message = prompt[start_idx:]
            return Llama2ChatResponse(model, tokenizer, length, seed, system_message, user_message)
        elif emails.BLOCK_OPT in prompt: # generate scenarios
            return Llama2ChatResponse(model, tokenizer, length, seed, "", prompt)
        else:
            raise NotImplementedError("Prompt not recognized!\n\n" + prompt)

    def score(self, model, tokenizer, prompt, outputs, seed) -> List[Llama2ChatScore]:
        log_probs = evaluate_predefined_outputs(model, tokenizer, prompt, outputs, seed)
        return [Llama2ChatScore(logprob=log_prob) for log_prob in log_probs]

    def list_models(self) -> List[Llama2ChatModel]:
        # llama2_models = ["/home/ewanlee/Models/llama-2-7b-chat/models/"]
        llama2_models = ["/home/ewanlee/Models/llama-3-8b-instruct/models/"]
        models = [Llama2ChatModel(model_name) for model_name in llama2_models]
        return models


class Llama2Chat:
    """Llama2Chat LLM."""

    def __init__(self, path=PATH):
        self.client = Llama2ChatClient()
        self.llm = Llama2ChatModel(path=path)

    def generate_response(self, prompt: str, seed: int, num_output_tokens: int) -> str:
        response = self.client.sample(
            model=self.llm.model, 
            tokenizer=self.llm.tokenizer, 
            length=num_output_tokens, 
            seed=seed, 
            prompt=prompt
        )
        return response.text

    def generate_bool(self, prompt: str, seed: int) -> bool:
        # prompt = 'Read the following summary of a dialgoue between two parties attempting to reach a trade agreement. Have the players reached a trade agreement? If a trade has been accepted or the players cannot come to an agreement, respond Yes. Otherwise, if the players are still discussing terms, respond No.Here is the dialogue:\n\nBob proposes trading one banana for one apple with Suzy. Suzy agrees.\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&Response: '
        response = self.client.sample(
            model=self.llm.model, 
            tokenizer=self.llm.tokenizer, 
            length=200, 
            seed=seed, 
            prompt=prompt
        )
        terminate = response.text.strip() == "Yes"
        return terminate
        # TODO: score comparison is not working
        # scores = self.client.score(model=self.llm.model, tokenizer=self.llm.tokenizer, prompt=prompt, outputs=["Yes", "No"], seed=seed)
        # score_true = scores[0].logprob
        # score_false = scores[1].logprob
        # if score_true > score_false:
        #     return True
        # else:
        #     return False
        

@dataclasses.dataclass(frozen=True)
class MockScore:
    logprob: float


class MockModel:
    """Mock LLM model."""

    def __init__(self, name):
        self.name = name


class MockResponse:
    """Mock LLM response."""

    def __init__(self, text):
        self.text = text


class MockClient:
    """Mock LLM client."""

    def __init__(self):
        # for cycling through mock response options
        self._idxs = {"names": 0, "tones": 0, "examples": 0}

    def sample(self, model: str, length: int, seed: int, prompt: str) -> MockResponse:
        """Returns string responses according to fixed prompt styles."""
        del model, length, seed
        prompt_lower = prompt.lower()
        if "names" in prompt_lower:
            dummy_names = ["Suzy", "Bob", "Alice", "Doug", "Arun", "Maria", "Zhang"]
            dummy_name = dummy_names[self._idxs["names"]]
            self._idxs["names"] = (self._idxs["names"] + 1) % len(dummy_names)
            return MockResponse(dummy_name + "\n")
        elif "tones" in prompt_lower:
            dummy_tones = ["Happy", "Sad", "Angry"]
            dummy_tone = dummy_tones[self._idxs["tones"]]
            self._idxs["tones"] = (self._idxs["tones"] + 1) % len(dummy_tones)
            return MockResponse(dummy_tone + "\n")
        elif "list of items" in prompt_lower:
            num_examples = 10
            dummy_examples = [f"Example-{i}" for i in range(num_examples)]
            dummy_example = dummy_examples[self._idxs["examples"]]
            self._idxs["examples"] = (self._idxs["examples"] + 1) % num_examples
            return MockResponse(dummy_example + "\n")
        elif "score" in prompt_lower or "value" in prompt_lower:
            return MockResponse("5\n")
        elif "summary" in prompt_lower:
            return MockResponse("This is a summary of the dialogue. We are happy.\n")
        elif emails.BLOCK_OPT in prompt:
            return MockResponse("\nThat all sounds good to me.\n")
        else:
            raise ValueError("Prompt not recognized!\n\n" + prompt)

    def score(self, model: str, prompt: str) -> List[MockScore]:
        del model, prompt
        return [MockScore(logprob=-1)]

    def list_models(self) -> List[MockModel]:
        dummy_models = ["dummy_model"]
        models = [MockModel(model_name) for model_name in dummy_models]
        return models


class MockLLM:
    """Mock LLM."""

    def __init__(self):
        self.client = MockClient()
        self.model = "dummy_model"

    def generate_response(self, prompt: str, seed: int, num_output_tokens: int) -> str:
        response = self.client.sample(
            model=self.model, length=num_output_tokens, seed=seed, prompt=prompt
        )
        return response.text

    def generate_bool(self, prompt: str, seed: int) -> bool:
        del seed
        score_true = self.client.score(model=self.model, prompt=prompt + "Yes")
        score_false = self.client.score(model=self.model, prompt=prompt + "No")
        if score_true > score_false:
            return True
        else:
            return False


class MockTokenizer:
    """Mock Tokenizer."""

    def to_int(self, text: str) -> np.ndarray:
        return np.zeros(len(text), dtype=np.int32)


class MockVectorizer:
    """Mock Vectorizer."""

    def __init__(self):
        self.tokenizer = MockTokenizer()

    def vectorize(self, text: str, obs_size: int) -> np.ndarray:
        observation = self.tokenizer.to_int(text)[:obs_size]
        num_pad = max(0, obs_size - observation.size)
        observation = np.pad(observation, (0, num_pad))
        return observation
    
class Llama2ChatModelTest:
    """Llama2Chat model."""

    def __init__(self, path: str):
        device_map = "cuda:0" if torch.cuda.is_available() else "auto"
        model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            # torch_dtype=torch.float16, # for llama 2
            torch_dtype=torch.bfloat16, # for llama 3
            # load_in_8bit=True,
            trust_remote_code=True,
            # use_flash_attention_2=True,
        )
        self.model = model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False)
        # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id # for llama 2
        self.tokenizer.use_default_system_prompt = True
        self.conversation_history = []

        system_prompt = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
        self.conversation_history.append({"role": "system", "content": system_prompt})

    def generate_response(self, prompt: str, seed: int, num_output_tokens: int) -> str:
        self.conversation_history.append({"role": "user", "content": prompt})
        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history, 
            add_generation_prompt=True, # for llama 3
            return_tensors="pt", 
            add_special_tokens=False)
        start_idx = input_ids.shape[-1]

        # for llama 3
        terminators = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        if torch.cuda.is_available():
            input_ids = input_ids.to("cuda")

        generate_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": num_output_tokens,
            "do_sample": True,
            # "top_k": 50, # for llama 2
            "top_p": 0.9,
            "temperature": 0.6,
            # "repetition_penalty": 1.2, # for llama 2
            # "num_beams": 1, # for llama 2
            "eos_token_id": terminators, # for llama 3
        }
        set_seed(seed)
        generate_ids = self.model.generate(**generate_kwargs)
        assistant_response = self.tokenizer.decode(generate_ids[0][start_idx:], skip_special_tokens=True)
        self.conversation_history.append({"role": "assistant", "content": assistant_response})
        print(assistant_response)
        print(self.tokenizer.apply_chat_template(self.conversation_history, tokenize=False))

if __name__ == "__main__":
    # Test Llama2Chat model
    llama2chat = Llama2ChatModelTest(path=PATH)
    prompt = "Hello, how are you?"
    seed = 1234
    num_output_tokens = 200
    llama2chat.generate_response(prompt, seed, num_output_tokens)
