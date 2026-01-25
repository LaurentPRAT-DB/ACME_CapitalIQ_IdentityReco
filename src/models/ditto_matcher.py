"""
Ditto entity matcher - fine-tuned BERT for entity matching
"""
from __future__ import annotations

import os
import pandas as pd
import torch
from typing import Tuple, Optional, List, Dict
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class DittoDataset(Dataset):
    """Dataset for Ditto entity pair classification"""

    def __init__(self, pairs: List[Tuple[str, str]], labels: List[int], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left, right = self.pairs[idx]
        label = self.labels[idx]

        # Tokenize the pair
        encoding = self.tokenizer(
            left,
            right,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }


class DittoMatcher:
    """Ditto fine-tuned matcher for entity pair classification"""

    def __init__(
        self,
        base_model: str = "distilbert-base-uncased",
        max_length: int = 256,
        device: str = None
    ):
        """
        Initialize Ditto matcher

        Args:
            base_model: Base transformer model
            max_length: Maximum sequence length
            device: Device to run on (cuda/cpu)
        """
        self.base_model = base_model
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Initialize tokenizer and model (will be loaded during train/load)
        self.tokenizer = None
        self.model = None

    def train(
        self,
        training_data_path: str,
        output_path: str,
        epochs: int = 20,
        batch_size: int = 64,
        learning_rate: float = 3e-5,
        val_split: float = 0.2
    ):
        """
        Fine-tune Ditto model on entity pairs

        Args:
            training_data_path: Path to CSV with training pairs
            output_path: Path to save trained model
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            val_split: Validation split ratio
        """
        print(f"Loading training data from {training_data_path}")
        df = pd.read_csv(training_data_path)

        # Validate format
        if not all(col in df.columns for col in ["left_entity", "right_entity", "label"]):
            raise ValueError("Training data must have columns: left_entity, right_entity, label")

        # Split into train/val
        val_size = int(len(df) * val_split)
        train_df = df[:-val_size] if val_size > 0 else df
        val_df = df[-val_size:] if val_size > 0 else None

        print(f"Training samples: {len(train_df)}")
        if val_df is not None:
            print(f"Validation samples: {len(val_df)}")

        # Initialize tokenizer and model
        print(f"Initializing model: {self.base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            num_labels=2  # Binary classification
        ).to(self.device)

        # Create datasets
        train_pairs = list(zip(train_df["left_entity"], train_df["right_entity"]))
        train_labels = train_df["label"].tolist()
        train_dataset = DittoDataset(train_pairs, train_labels, self.tokenizer, self.max_length)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)

        # Training loop
        print(f"Training for {epochs} epochs...")
        self.model.train()

        for epoch in range(epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress_bar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                total_loss += loss.item()

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress_bar.set_postfix({"loss": loss.item()})

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1} - Avg Loss: {avg_loss:.4f}")

        # Save model
        self.save_model(output_path)
        print(f"Model saved to {output_path}")

    def predict(
        self,
        left_entity: str,
        right_entity: str
    ) -> Tuple[int, float]:
        """
        Predict if two entities match

        Args:
            left_entity: First entity (Ditto format)
            right_entity: Second entity (Ditto format)

        Returns:
            (prediction, confidence) - prediction is 0 or 1, confidence is probability
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()

        # Tokenize
        encoding = self.tokenizer(
            left_entity,
            right_entity,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)

        # Predict
        with torch.no_grad():
            outputs = self.model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

            prediction = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][prediction].item()

        return prediction, confidence

    def predict_batch(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: int = 32
    ) -> List[Tuple[int, float]]:
        """
        Predict matches for a batch of entity pairs

        Args:
            pairs: List of (left_entity, right_entity) tuples
            batch_size: Batch size for inference

        Returns:
            List of (prediction, confidence) tuples
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        self.model.eval()
        results = []

        # Process in batches
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]

            # Tokenize batch
            left_texts = [p[0] for p in batch_pairs]
            right_texts = [p[1] for p in batch_pairs]

            encodings = self.tokenizer(
                left_texts,
                right_texts,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)

            # Predict
            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

                predictions = torch.argmax(probs, dim=-1).cpu().numpy()
                confidences = probs.cpu().numpy()

                # Extract confidence for predicted class
                for j, pred in enumerate(predictions):
                    conf = confidences[j][pred]
                    results.append((int(pred), float(conf)))

        return results

    def save_model(self, output_path: str):
        """Save trained model"""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        print(f"Model saved to {output_path}")

    def load_model(self, model_path: str):
        """Load trained model"""
        print(f"Loading model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

    def evaluate(self, test_data_path: str) -> Dict[str, float]:
        """
        Evaluate model on test data

        Args:
            test_data_path: Path to CSV with test pairs

        Returns:
            Dictionary with metrics (accuracy, precision, recall, f1)
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.read_csv(test_data_path)
        pairs = list(zip(df["left_entity"], df["right_entity"]))
        true_labels = df["label"].tolist()

        # Predict
        predictions = self.predict_batch(pairs)
        pred_labels = [p[0] for p in predictions]

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(true_labels, pred_labels),
            "precision": precision_score(true_labels, pred_labels),
            "recall": recall_score(true_labels, pred_labels),
            "f1_score": f1_score(true_labels, pred_labels)
        }

        return metrics


def train_cli():
    """
    CLI entry point for training Ditto model
    Used by Databricks Asset Bundle python_wheel_task
    """
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Train Ditto entity matching model"
    )
    parser.add_argument(
        "--training-data",
        required=True,
        help="Path to training data CSV file"
    )
    parser.add_argument(
        "--output-path",
        required=True,
        help="Path to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs (default: 20)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Training batch size (default: 64)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-5,
        help="Learning rate (default: 3e-5)"
    )
    parser.add_argument(
        "--base-model",
        default="distilbert-base-uncased",
        help="Base transformer model (default: distilbert-base-uncased)"
    )

    args = parser.parse_args()

    print("="*80)
    print("Ditto Model Training")
    print("="*80)
    print(f"Training data: {args.training_data}")
    print(f"Output path: {args.output_path}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Base model: {args.base_model}")
    print("="*80)

    try:
        # Initialize matcher
        matcher = DittoMatcher(base_model=args.base_model)

        # Train model
        matcher.train(
            training_data_path=args.training_data,
            output_path=args.output_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )

        print("="*80)
        print("✓ Training completed successfully!")
        print(f"Model saved to: {args.output_path}")
        print("="*80)

        return 0

    except Exception as e:
        print("="*80)
        print(f"✗ Training failed: {str(e)}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    """Allow running as: python -m src.models.ditto_matcher"""
    import sys
    sys.exit(train_cli())
