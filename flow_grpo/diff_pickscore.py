import torch
import torch.nn as nn
import torch.nn.functional as F

def _clip_preprocess(images, size, mean, std):
    x = F.interpolate(images, (size, size), mode="bicubic", align_corners=False)
    return (x - mean.to(dtype=x.dtype)) / std.to(dtype=x.dtype)

class DifferentiablePickScore(nn.Module):
    def __init__(
        self,
        device,
        pickscore_model: str = "yuvalkirstain/PickScore_v1",
        clip_processor: str = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    ):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(clip_processor)
        self.model = CLIPModel.from_pretrained(pickscore_model).eval().to(device)
        self.model.requires_grad_(False)
        ip = self.processor.image_processor
        self.register_buffer(
            "mean", torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std", torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1),
        )
        self.size = int(ip.crop_size["height"])

    def forward(self, images, prompts, image_grad: bool = False):
        # Images Numerical Range: [0, 1]
        with torch.no_grad():
            tokens = self.processor(text=list(prompts), padding=True, truncation=True,
                                    max_length=77, return_tensors="pt")
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            text_embeds = F.normalize(self.model.get_text_features(**tokens), dim=-1)
            logit_scale = self.model.logit_scale.exp()

        if image_grad:
            with torch.enable_grad():
                pixels = _clip_preprocess(images, self.size, self.mean, self.std)
                image_embeds = F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
                scores = logit_scale * (image_embeds * text_embeds).sum(-1) / 26.0
                return scores
        else:
            with torch.no_grad():
                pixels = _clip_preprocess(images, self.size, self.mean, self.std)
                image_embeds = F.normalize(self.model.get_image_features(pixel_values=pixels), dim=-1)
                return logit_scale * (image_embeds * text_embeds).sum(-1) / 26.0