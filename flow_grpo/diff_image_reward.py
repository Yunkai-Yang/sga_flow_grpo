import torch
import torch.nn as nn
import torch.nn.functional as F

def _clip_preprocess(images, size, mean, std):
    x = F.interpolate(images, (size, size), mode="bicubic", align_corners=False)
    return (x - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)

class DifferentiableImageReward(nn.Module):
    def __init__(
        self,
        device,
        model_id: str = "ImageReward-v1.0",
        model_root: str | None = None
    ):
        super().__init__()
        import ImageReward as RM
        self.device = device
        self.model = RM.load(model_id, device=device, download_root=model_root).eval()

        self.model.requires_grad_(False)
        self.register_buffer(
            "mean", torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device
        ).view(1, 3, 1, 1))
        self.register_buffer(
            "std", torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device
        ).view(1, 3, 1, 1))
        self.size = 224

    def _score_tensor(self, images, tokens):
        pixels = _clip_preprocess(images, self.size, self.mean, self.std)
        image_embeds = self.model.blip.visual_encoder(pixels)
        image_atts = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=image_embeds.device
        )
        text_output = self.model.blip.text_encoder(
            tokens.input_ids,
            attention_mask=tokens.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        text_features = text_output.last_hidden_state[:, 0, :].float()
        rewards = self.model.mlp(text_features)
        rewards = (rewards - self.model.mean) / self.model.std
        return rewards.flatten()

    def forward(self, images, prompts, image_grad: bool = False):

        with torch.no_grad():
            tokens = self.model.blip.tokenizer(
                list(prompts), padding="max_length", truncation=True,
                max_length=35, return_tensors="pt"
            ).to(self.device)

        if image_grad:
            with torch.enable_grad():
                scores = self._score_tensor(images, tokens)
                return scores
        with torch.no_grad():
            return self._score_tensor(images, tokens)