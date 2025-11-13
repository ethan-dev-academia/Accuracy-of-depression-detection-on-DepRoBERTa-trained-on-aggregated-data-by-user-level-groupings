
"""Step-by-step helper for `rafalposwiata/deproberta-large-depression`.

The script mirrors the numbered guide provided in the prompt and exposes each
stage (load model, ingest local data, tokenize, optionally fine-tune, run
inference) as an interactive step. By default it pauses before every action so
you can review the plan and decide whether to continue.

Use `--auto` to run the steps without confirmation. Supply `--train` only if
your data includes ground-truth labels; otherwise the script will skip the
training stages and instead demonstrate inference on the aggregated users.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import torch
from datasets import Dataset
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

LOGGER = logging.getLogger(__name__)

REMOVED_MARKERS = {
    "[removed]",
    "[deleted]",
    "removed",
    "deleted",
    "user deleted",
    "removed by moderator",
    "removed by automoderator",
    "[removed by moderator]",
    "[deleted by user]",
}


def is_significant_text(text: str) -> bool:
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return False

    simplified = cleaned.lower()
    simplified = simplified.replace("\u2019", "'").strip()

    if simplified in REMOVED_MARKERS:
        return False
    if simplified.startswith("[removed"):
        return False
    if simplified.startswith("[deleted"):
        return False

    alnum_ratio = sum(ch.isalnum() for ch in cleaned) / max(len(cleaned), 1)
    if alnum_ratio < 0.2:
        return False

    return True


def load_user_records(dataset_path: str, limit: Optional[int] = None) -> List[Dict]:
    """Load user-level JSON records from a file or directory."""
    path = Path(dataset_path)

    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.json"))
        if not files:
            raise FileNotFoundError(
                f"No JSON files found under directory: {dataset_path}"
            )
    else:
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    records: List[Dict] = []
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            LOGGER.warning("Skipping %s due to JSON error: %s", file_path, exc)
            continue

        if isinstance(payload, list):
            records.extend(payload)
        elif isinstance(payload, dict):
            records.append(payload)
        else:
            LOGGER.warning("Unhandled JSON root type in %s: %s", file_path, type(payload))

        if limit and len(records) >= limit:
            records = records[:limit]
            break

    LOGGER.info("Loaded %d user records from %s file(s)", len(records), len(files))
    return records


@dataclass
class AggregatedDatasets:
    dataset_all: Dataset
    dataset_labeled: Optional[Dataset]
    label_counts: Counter
    stats: Dict[str, int]


def aggregate_user_posts(
    records: Sequence[Dict],
    label_field: Optional[str] = None,
) -> AggregatedDatasets:
    """Aggregate user posts/comments, producing datasets for inference and optional training."""

    user_ids_all: List[str] = []
    texts_all: List[str] = []
    segments_all: List[List[str]] = []
    labels_all: List[Optional[int]] = []

    labeled_user_ids: List[str] = []
    labeled_texts: List[str] = []
    labeled_segments: List[List[str]] = []
    labeled_labels: List[int] = []

    empty_text_users = 0
    label_counts: Counter = Counter()

    for record in records:
        user_id = str(record.get("username") or record.get("user_id") or "unknown")
        posts = record.get("posts") or []
        comments = record.get("comments") or []

        candidate_segments: List[tuple[int, int, str]] = []

        def add_candidate(raw_text: str, score: Optional[int]) -> None:
            if not raw_text:
                return
            cleaned = " ".join(raw_text.split()).strip()
            if not is_significant_text(cleaned):
                return
            candidate_segments.append(
                (
                    int(score or 0),
                    len(cleaned),
                    cleaned,
                )
            )

        for post in posts:
            if not isinstance(post, dict):
                continue
            title = post.get("title") or ""
            content = post.get("content") or ""
            score = post.get("score") or 0

            combined_parts: List[str] = []
            if is_significant_text(title):
                combined_parts.append(title.strip())
            if is_significant_text(content):
                combined_parts.append(content.strip())

            if combined_parts:
                combined_text = " ".join(combined_parts)
                add_candidate(combined_text, score)

        for comment in comments:
            if not isinstance(comment, dict):
                continue
            body = comment.get("content") or comment.get("body") or ""
            score = comment.get("score") or 0
            add_candidate(body, score)

        if not candidate_segments:
            empty_text_users += 1
            continue

        candidate_segments.sort(key=lambda item: (item[0], item[1]), reverse=True)

        seen_segments = set()
        cleaned_segments: List[str] = []
        for _, _, segment in candidate_segments:
            if segment in seen_segments:
                continue
            cleaned_segments.append(segment)
            seen_segments.add(segment)

        if not cleaned_segments:
            empty_text_users += 1
            continue

        combined_text = " ".join(cleaned_segments)

        user_ids_all.append(user_id)
        texts_all.append(combined_text)
        segments_all.append(cleaned_segments)

        label_value: Optional[int] = None
        if label_field and record.get(label_field) is not None:
            try:
                label_value = int(record[label_field])
            except (TypeError, ValueError):
                LOGGER.warning(
                    "Skipping non-integer label %r for user %s", record[label_field], user_id
                )
            else:
                labeled_user_ids.append(user_id)
                labeled_texts.append(combined_text)
                labeled_segments.append(cleaned_segments)
                labeled_labels.append(label_value)
                label_counts[label_value] += 1

        labels_all.append(label_value)

    dataset_all = Dataset.from_dict(
        {"user_id": user_ids_all, "text": texts_all, "segments": segments_all, "label": labels_all}
    )
    dataset_labeled: Optional[Dataset] = None
    if labeled_labels:
        dataset_labeled = Dataset.from_dict(
            {
                "user_id": labeled_user_ids,
                "text": labeled_texts,
                "segments": labeled_segments,
                "label": labeled_labels,
            }
        )

    stats = {
        "total_records": len(records),
        "users_with_aggregated_text": len(dataset_all),
        "users_with_labels": len(labeled_labels),
        "users_without_text": empty_text_users,
    }

    LOGGER.info(
        "Aggregated %d users with text (%d without text skipped); %d users have labels.",
        stats["users_with_aggregated_text"],
        stats["users_without_text"],
        stats["users_with_labels"],
    )

    if label_counts:
        LOGGER.info("Label distribution: %s", dict(label_counts.most_common()))
    else:
        LOGGER.info("No labels found in the provided records.")

    return AggregatedDatasets(
        dataset_all=dataset_all,
        dataset_labeled=dataset_labeled,
        label_counts=label_counts,
        stats=stats,
    )


def tokenize_dataset(dataset: Optional[Dataset], tokenizer) -> Optional[Dataset]:
    """Step 5: Tokenize labeled dataset and create train/test split."""

    if dataset is None or len(dataset) == 0:
        LOGGER.info("No labeled dataset available; skipping tokenization.")
        return None

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    tokenized = dataset.map(tokenize_fn, batched=True)
    tokenized = tokenized.train_test_split(test_size=0.1, seed=42)
    removable_columns = [
        column
        for column in tokenized["train"].column_names
        if column in {"text", "user_id", "segments"}
    ]
    if removable_columns:
        tokenized = tokenized.remove_columns(removable_columns)
    LOGGER.info(
        "Tokenized dataset split into %d train and %d test samples",
        len(tokenized["train"]),
        len(tokenized["test"]),
    )
    return tokenized


def compute_metrics(pred):
    """Step 6: Compute evaluation metrics."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def build_trainer(tokenized_dataset, tokenizer, model, output_dir: str):
    """Step 7 & 8: Create `TrainingArguments` and `Trainer`."""
    if tokenized_dataset is None:
        raise ValueError("Tokenized dataset is required to build a Trainer.")
    signature_params = inspect.signature(TrainingArguments).parameters
    strategy_key = (
        "evaluation_strategy"
        if "evaluation_strategy" in signature_params
        else "eval_strategy"
    )

    training_args_kwargs = {
        "output_dir": output_dir,
        "save_strategy": "epoch",
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "logging_dir": "./logs",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "logging_steps": 10,
        "report_to": "none",
    }
    training_args_kwargs[strategy_key] = "epoch"

    training_args = TrainingArguments(**training_args_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    return trainer


def load_model_and_tokenizer(model_name: str):
    """Step 3: Load tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactive helper for fine-tuning rafalposwiata/deproberta-large-depression "
            "on user-level text."
        )
    )
    parser.add_argument(
        "--dataset-path",
        default="F:\\DATA STORAGE\\AGG_PACKET",
        help="Path to a JSON file or directory containing user-level Reddit exports.",
    )
    parser.add_argument(
        "--label-field",
        default=None,
        help="Optional key in each user record that stores the target label. "
        "If omitted or absent, training-related steps are skipped.",
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=500,
        help="Limit the number of user records loaded from disk (use 0 for no limit).",
    )
    parser.add_argument(
        "--demo-sample-size",
        type=int,
        default=5,
        help="How many users to include in the post-training inference demo.",
    )
    parser.add_argument(
        "--demo-message-count",
        type=int,
        default=3,
        help="Number of individual messages/comments to display per demo user.",
    )
    parser.add_argument(
        "--demo-message-chars",
        type=int,
        default=200,
        help="Maximum number of characters to show for each demo message.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Enable the training/evaluation/save stages (Steps 9-10).",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run all available steps without pausing for confirmation.",
    )
    parser.add_argument(
        "--model-name",
        default="rafalposwiata/deproberta-large-depression",
        help="Model checkpoint to load.",
    )
    parser.add_argument(
        "--output-dir",
        default="./ModelB_final",
        help="Directory used when saving the fine-tuned model and tokenizer.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity for logging output.",
    )
    return parser.parse_args()


@dataclass
class PipelineStep:
    name: str
    description: str
    handler: Callable[[argparse.Namespace, dict], None]
    requires_train: bool = False


def step_load(args: argparse.Namespace, context: dict) -> None:
    LOGGER.info("Step 3: Loading tokenizer and model from %s", args.model_name)
    tokenizer, model = load_model_and_tokenizer(args.model_name)
    context["tokenizer"] = tokenizer
    context["model"] = model


def step_aggregate(args: argparse.Namespace, context: dict) -> None:
    LOGGER.info("Step 4: Loading and aggregating user data from %s", args.dataset_path)

    limit = args.max_users if args.max_users > 0 else None
    records = load_user_records(args.dataset_path, limit=limit)
    context["raw_records"] = records

    aggregated = aggregate_user_posts(records, label_field=args.label_field)
    context["dataset_all"] = aggregated.dataset_all
    context["dataset_labeled"] = aggregated.dataset_labeled
    context["label_counts"] = aggregated.label_counts
    context["aggregation_stats"] = aggregated.stats

    LOGGER.info("Aggregation stats: %s", aggregated.stats)
    if aggregated.dataset_labeled is None or len(aggregated.dataset_labeled) == 0:
        LOGGER.info(
            "No labeled examples detected. Training-related steps will be skipped unless labels are provided."
        )


def step_tokenize(args: argparse.Namespace, context: dict) -> None:
    LOGGER.info("Step 5: Tokenizing dataset and creating train/test split")
    if "tokenizer" not in context:
        raise RuntimeError("Tokenize step requires tokenizer from previous steps.")

    labeled_dataset = context.get("dataset_labeled")
    if labeled_dataset is None or len(labeled_dataset) == 0:
        LOGGER.info("No labeled dataset available. Skipping tokenization step.")
        context["tokenized_dataset"] = None
        return

    context["tokenized_dataset"] = tokenize_dataset(labeled_dataset, context["tokenizer"])


def step_build_trainer(args: argparse.Namespace, context: dict) -> None:
    LOGGER.info("Steps 7-8: Creating training arguments and Trainer instance")
    if "model" not in context or "tokenizer" not in context:
        raise RuntimeError("Trainer step requires tokenizer and model.")

    tokenized_dataset = context.get("tokenized_dataset")
    if tokenized_dataset is None:
        LOGGER.info("Tokenized dataset missing; skipping trainer construction.")
        context["trainer"] = None
        return

    context["trainer"] = build_trainer(
        tokenized_dataset,
        context["tokenizer"],
        context["model"],
        output_dir=args.output_dir,
    )


def step_train_and_eval(args: argparse.Namespace, context: dict) -> None:
    trainer: Optional[Trainer] = context.get("trainer")
    if trainer is None:
        LOGGER.info("Trainer not available; skipping training and evaluation.")
        return

    LOGGER.info("Step 9: Training the model on the prepared dataset")
    trainer.train()

    LOGGER.info("Evaluating best checkpoint on the validation split")
    results = trainer.evaluate()
    context["results"] = results
    LOGGER.info("Evaluation metrics: %s", results)


def step_save(args: argparse.Namespace, context: dict) -> None:
    trainer: Optional[Trainer] = context.get("trainer")
    tokenizer = context.get("tokenizer")

    if trainer is None or tokenizer is None:
        LOGGER.info("Trainer or tokenizer missing; skipping save step.")
        return

    LOGGER.info("Step 10: Saving model and tokenizer to %s", args.output_dir)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


def step_demo_inference(args: argparse.Namespace, context: dict) -> None:
    tokenizer = context.get("tokenizer")
    if tokenizer is None:
        LOGGER.info("Tokenizer missing; skipping demo inference.")
        return

    trainer: Optional[Trainer] = context.get("trainer")
    model = trainer.model if trainer is not None else context.get("model")
    if model is None:
        LOGGER.info("Model not available; skipping demo inference.")
        return

    dataset_all: Optional[Dataset] = context.get("dataset_all")
    if dataset_all is None or len(dataset_all) == 0:
        LOGGER.info("Aggregated dataset empty; skipping demo inference.")
        return

    LOGGER.info("Step 11: Running example inference to inspect predictions")
    model.eval()

    sample_size = min(args.demo_sample_size, len(dataset_all))
    sample_dataset = dataset_all.select(range(sample_size))

    predictions_counter: Counter = Counter()

    for row in sample_dataset:
        text = row["text"]
        user_id = row.get("user_id", "unknown")
        segments = row.get("segments") or []
        true_label = row.get("label")
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=-1).cpu().tolist()[0]
            predicted_label = int(torch.argmax(logits, dim=-1).item())

        id2label = getattr(getattr(model, "config", None), "id2label", None) or {
            idx: f"LABEL_{idx}" for idx in range(logits.shape[-1])
        }

        predictions_counter[predicted_label] += 1

        class_probabilities = sorted(
            [
                (id2label.get(idx, f"LABEL_{idx}"), round(prob * 100, 2))
                for idx, prob in enumerate(probs)
            ],
            key=lambda item: item[1],
            reverse=True,
        )

        LOGGER.info("User: %s", user_id)
        LOGGER.info(
            "Predicted class: %s (id=%d)",
            id2label.get(predicted_label, predicted_label),
            predicted_label,
        )
        if segments:
            LOGGER.info("Sample messages:")
            for idx, segment in enumerate(segments[: args.demo_message_count], start=1):
                trimmed = segment.strip().replace("\n", " ")
                if len(trimmed) > args.demo_message_chars:
                    trimmed = trimmed[: args.demo_message_chars].rstrip() + "..."
                LOGGER.info("")
                LOGGER.info("  [%d] %s", idx, trimmed)
            LOGGER.info("")
        LOGGER.info(
            "Class probabilities (percent): %s",
            ", ".join(f"{label}: {score}%" for label, score in class_probabilities),
        )

        if true_label is not None:
            LOGGER.info(
                "True label: %s (id=%s)",
                id2label.get(true_label, str(true_label)),
                true_label,
            )
        else:
            LOGGER.info("True label: unavailable for this user.")

        LOGGER.debug("Raw logits: %s", logits.tolist())

    if predictions_counter:
        id2label = getattr(getattr(model, "config", None), "id2label", None) or {
            idx: f"LABEL_{idx}" for idx in range(model.config.num_labels)
        }
        distribution = {
            id2label.get(label_id, str(label_id)): count
            for label_id, count in predictions_counter.items()
        }
        LOGGER.info(
            "Prediction distribution across %d sampled users: %s",
            sample_size,
            distribution,
        )


def should_run_step(step: PipelineStep, step_index: int, total_steps: int, auto_mode: bool) -> bool:
    if auto_mode:
        return True

    prompt = (
        f"[Step {step_index}/{total_steps}] {step.description}\n"
        "Proceed? [y/N]: "
    )
    response = input(prompt).strip().lower()
    return response in {"y", "yes"}


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    LOGGER.info("PyTorch sees CUDA: %s", torch.cuda.is_available())

    context: dict = {}

    steps = [
        PipelineStep(
            name="load_model",
            description="Load pretrained model and tokenizer (Step 3)",
            handler=step_load,
        ),
        PipelineStep(
            name="aggregate_data",
            description="Aggregate user posts/comments from local JSON (Step 4)",
            handler=step_aggregate,
        ),
        PipelineStep(
            name="tokenize",
            description="Tokenize labeled data and create train/test split (Step 5)",
            handler=step_tokenize,
        ),
        PipelineStep(
            name="build_trainer",
            description="Configure TrainingArguments and instantiate Trainer (Steps 7-8)",
            handler=step_build_trainer,
        ),
        PipelineStep(
            name="train_and_eval",
            description="Fine-tune the model and evaluate on validation set (Step 9)",
            handler=step_train_and_eval,
            requires_train=True,
        ),
        PipelineStep(
            name="save_artifacts",
            description="Save the fine-tuned model and tokenizer (Step 10)",
            handler=step_save,
            requires_train=True,
        ),
        PipelineStep(
            name="demo_inference",
            description="Run a quick inference example to inspect predictions (Step 11)",
            handler=step_demo_inference,
        ),
    ]

    total_steps = len(steps)

    for index, step in enumerate(steps, start=1):
        if step.requires_train and not args.train:
            LOGGER.info(
                "Skipping %s because --train was not provided. Re-run with --train to include this step.",
                step.name,
            )
            continue

        LOGGER.debug("Preparing to run step: %s", step.name)

        if not should_run_step(step, index, total_steps, args.auto):
            LOGGER.info("Skipping step: %s", step.name)
            continue

        LOGGER.info("Running step: %s", step.description)
        step.handler(args, context)
        LOGGER.info("Completed step: %s\n", step.name)

    LOGGER.info("Pipeline complete. Context keys available: %s", sorted(context.keys()))


if __name__ == "__main__":
    main()


