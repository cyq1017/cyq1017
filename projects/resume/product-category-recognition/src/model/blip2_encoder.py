"""
BLIP2 Model Encoder wrapper.

Provides unified API for extracting text, image, and multimodal embeddings
using the BLIP2 Q-Former architecture.

Compatible with transformers >= 5.x where get_text_features and
get_qformer_features were removed.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Encoder:
    """
    BLIP2 encoder for extracting text, image, and multimodal embeddings.

    Uses Q-Former to produce unified representations across modalities.
    For image embeddings: vision_model → Q-Former → language_projection.
    For text embeddings: tokenize → Q-Former (query_tokens as query).
    For multimodal: vision_model → Q-Former (cross-attend to image) → language_projection.
    """

    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b",
                 device: str = "cuda", dtype: str = "float16",
                 lora_adapter_path: str | None = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

        print(f"Loading BLIP2 model: {model_name} on {self.device} ({dtype})")
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        if lora_adapter_path is not None:
            from peft import PeftModel
            print(f"Loading LoRA adapter: {lora_adapter_path}")
            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
            self.model.eval()

        print("BLIP2 model loaded.")

    def _get_vision_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run vision encoder and return image embeddings [B, seq_len, hidden]."""
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values,
            return_dict=True,
        )
        return vision_outputs.last_hidden_state

    def _qformer_with_vision(self, image_embeds: torch.Tensor) -> torch.Tensor:
        """Run Q-Former with image cross-attention, return projected features [B, num_query, D]."""
        image_attn_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)

        qformer_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn_mask,
            return_dict=True,
        )
        query_output = qformer_outputs.last_hidden_state
        if query_output.dtype != image_embeds.dtype:
            query_output = query_output.to(image_embeds.dtype)

        return self.model.language_projection(query_output)

    @torch.no_grad()
    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Extract text embedding via LM input embeddings (same space as
        language_projection output). Returns [D] tensor.

        Since BLIP2's Q-Former requires image cross-attention, text-only
        embeddings use the LM word embedding layer directly. This works
        because language_projection maps Q-Former output INTO the LM
        embedding space, making them comparable via cosine similarity.
        """
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].to(self.device)

        text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        embedding = text_embeds.mean(dim=1).squeeze(0)
        return F.normalize(embedding.float(), dim=-1).cpu()

    @torch.no_grad()
    def get_image_embedding(self, image: Image.Image) -> torch.Tensor:
        """Extract image embedding via Q-Former. Returns [D] tensor."""
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        image_embeds = self._get_vision_embeds(pixel_values)
        projected = self._qformer_with_vision(image_embeds)

        # Mean pool over query tokens → [D]
        embedding = projected.mean(dim=1).squeeze(0)
        return F.normalize(embedding, dim=-1).cpu().float()

    @torch.no_grad()
    def get_multimodal_embedding(self, text: str, image: Image.Image) -> torch.Tensor:
        """
        Extract multimodal embedding via Q-Former (image-grounded).
        Returns [D] tensor.
        """
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

        image_embeds = self._get_vision_embeds(pixel_values)
        projected = self._qformer_with_vision(image_embeds)

        # Mean pool over query tokens → [D]
        embedding = projected.mean(dim=1).squeeze(0)
        return F.normalize(embedding, dim=-1).cpu().float()

    @torch.no_grad()
    def get_text_embeddings_batch(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        """Batch text embedding extraction via LM word embeddings. Returns [N, D] tensor."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]

            inputs = self.processor(text=batch_texts, return_tensors="pt",
                                    padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.device)

            text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            embeddings = text_embeds.mean(dim=1)  # [B, D]
            embeddings = F.normalize(embeddings.float(), dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def get_image_embeddings_batch(self, images: list[Image.Image],
                                   batch_size: int = 32) -> torch.Tensor:
        """Batch image embedding extraction. Returns [N, D] tensor."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            inputs = self.processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)

            image_embeds = self._get_vision_embeds(pixel_values)
            projected = self._qformer_with_vision(image_embeds)

            # Mean pool over query tokens → [B, D]
            embeddings = projected.mean(dim=1)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def get_multimodal_embeddings_batch(self, texts: list[str],
                                         images: list[Image.Image],
                                         batch_size: int = 16) -> torch.Tensor:
        """Batch multimodal embedding extraction. Returns [N, D] tensor."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_images = images[i:i + batch_size]

            # Image embedding via Q-Former
            inputs = self.processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(self.device, dtype=self.dtype)
            image_embeds = self._get_vision_embeds(pixel_values)
            projected = self._qformer_with_vision(image_embeds)
            img_emb = projected.mean(dim=1)  # [B, D]

            # Text embedding via LM word embeddings
            txt_inputs = self.processor(text=batch_texts, return_tensors="pt",
                                         padding=True, truncation=True)
            input_ids = txt_inputs["input_ids"].to(self.device)
            txt_emb = self.model.language_model.get_input_embeddings()(input_ids).mean(dim=1)

            # Weighted combination (image-dominant for multimodal)
            combined = 0.7 * img_emb.float() + 0.3 * txt_emb.float()
            embeddings = F.normalize(combined, dim=-1)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)
