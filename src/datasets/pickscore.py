import json
import os
import torch

from safetensors.torch import load_file
from torch.utils.data import Dataset

from ..utils.hash import hash_md5_trunc8

def load_text_embeds(vector_dir, prompt):
    prompt_md5 = hash_md5_trunc8(prompt)
    return load_file(os.path.join(vector_dir, f'{prompt_md5}.safetensors'))


class PickScorePromptDataset(Dataset):
    def __init__(self, dataset, split='train', vector_dir = None):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        self.metadatas = []
        self.prompts = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.prompts.append(line)
                self.metadatas.append({})

        self.vector_dir = vector_dir
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        if self.vector_dir:
            prompt_vector = load_text_embeds(self.vector_dir, prompt)
            return {"prompt": prompt, "metadata": self.metadatas[idx], **prompt_vector}
        else:
            return {"prompt": prompt, "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        if "prompt_embeds" in examples[0]:
            pooled_prompt_embeds = torch.cat([example['pooled_prompt_embeds'] for example in examples], dim=0)
            prompt_embeds = torch.cat([example['prompt_embeds'] for example in examples], dim=0)
            return prompts, metadatas, (pooled_prompt_embeds, prompt_embeds)
        else:
            return prompts, metadatas