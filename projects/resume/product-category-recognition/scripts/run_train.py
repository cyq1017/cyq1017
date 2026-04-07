# scripts/run_train.py
"""V2 LoRA training entry point."""
from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from peft import get_peft_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.train.dataset import ITCDataset
from src.train.lora_config import create_lora_config, load_train_config
from src.train.trainer import ITCTrainer


def collate_fn(batch):
    images, texts, labels = zip(*batch)
    return list(images), list(texts), list(labels)


def main():
    config = load_train_config()
    tc = config["training"]
    lc = config["lora"]

    print("Loading BLIP2 model...")
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
    ).cuda()

    # Apply LoRA to Q-Former ONLY (not LM or vision encoder)
    lora_config = create_lora_config(rank=lc["rank"], alpha=lc["alpha"], dropout=lc["dropout"])
    from src.train.lora_config import validate_target_modules
    validate_target_modules(model.qformer, lora_config.target_modules)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Dataset
    data_dir = PROJECT_ROOT / "data"
    dataset = ITCDataset(data_dir / "v2_train.csv",
                         data_dir / "category_descriptions.json")
    dataloader = DataLoader(dataset, batch_size=tc["batch_size"],
                            shuffle=True, collate_fn=collate_fn, num_workers=4)

    # Optimizer
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=tc["learning_rate"], weight_decay=tc["weight_decay"])
    total_steps = len(dataloader) * tc["epochs"] // tc["gradient_accumulation_steps"]
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Train
    trainer = ITCTrainer(model, processor, optimizer, scheduler,
                         fp16=tc["fp16"], grad_accum_steps=tc["gradient_accumulation_steps"])

    for epoch in range(1, tc["epochs"] + 1):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        print(f"Epoch {epoch}: avg_loss={avg_loss:.4f}")
        trainer.save_checkpoint(epoch)

    print("Training complete. Run evaluation with the saved adapter.")


if __name__ == "__main__":
    main()
