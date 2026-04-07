# src/train/trainer.py
"""ITC training loop for LoRA fine-tuning."""
from __future__ import annotations
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compute_itc_loss(image_embs: torch.Tensor, text_embs: torch.Tensor,
                     temperature: float = 0.07) -> torch.Tensor:
    """
    Compute symmetric Image-Text Contrastive loss.

    Args:
        image_embs: [B, D] normalized image embeddings
        text_embs: [B, D] normalized text embeddings
        temperature: scaling factor

    Returns:
        Scalar loss (average of image->text and text->image cross-entropy)
    """
    image_embs = F.normalize(image_embs, dim=-1)
    text_embs = F.normalize(text_embs, dim=-1)

    logits = image_embs @ text_embs.T / temperature  # [B, B]
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.T, labels)
    return (loss_i2t + loss_t2i) / 2


class ITCTrainer:
    """Trainer for ITC LoRA fine-tuning."""

    def __init__(self, model, processor, optimizer, scheduler=None,
                 device="cuda", fp16=True, grad_accum_steps=2,
                 checkpoint_dir="checkpoints"):
        self.model = model
        self.processor = processor
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.fp16 = fp16
        self.grad_accum_steps = grad_accum_steps
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_image_embedding(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Image -> Q-Former -> language_projection -> [B, D]"""
        dtype = torch.float16 if self.fp16 else torch.float32
        pixel_values = pixel_values.to(self.device, dtype=dtype)

        vision_outputs = self.model.vision_model(pixel_values=pixel_values, return_dict=True)
        image_embeds = vision_outputs.last_hidden_state
        image_attn = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=self.device)

        query_tokens = self.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        qformer_outputs = self.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attn,
            return_dict=True,
        )
        query_output = qformer_outputs.last_hidden_state.to(dtype)
        projected = self.model.language_projection(query_output)
        return F.normalize(projected.mean(dim=1).float(), dim=-1)

    def _get_text_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Text -> LM word_embeddings -> mean-pool -> [B, D] (frozen, same as V1)"""
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            text_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        return F.normalize(text_embeds.mean(dim=1).float(), dim=-1)

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_steps = 0

        for step, (images, texts, labels) in enumerate(dataloader):
            img_inputs = self.processor(images=list(images), return_tensors="pt")
            txt_inputs = self.processor(text=list(texts), return_tensors="pt",
                                        padding=True, truncation=True)

            image_embs = self._get_image_embedding(img_inputs["pixel_values"])
            text_embs = self._get_text_embedding(txt_inputs["input_ids"])

            loss = compute_itc_loss(image_embs, text_embs) / self.grad_accum_steps
            loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            n_steps += 1

            if (step + 1) % 100 == 0:
                avg = total_loss / n_steps
                print(f"  Epoch {epoch} Step {step+1}: loss={avg:.4f}")

        return total_loss / max(n_steps, 1)

    def save_checkpoint(self, epoch: int, tag: str = ""):
        """Save LoRA adapter weights."""
        name = f"epoch{epoch}" if not tag else tag
        save_path = self.checkpoint_dir / name
        self.model.save_pretrained(str(save_path))
        print(f"Checkpoint saved: {save_path}")
