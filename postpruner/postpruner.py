import argparse
import logging
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    set_seed,
)

from postpruner.dataset.glue import glue_dataset, max_seq_length, avg_seq_length
from postpruner.dataset.squad import squad_dataset
from postpruner.efficiency.mac import compute_mask_mac
from postpruner.efficiency.latency import estimate_latency
from postpruner.prune.fisher import collect_mask_grads
from postpruner.prune.search import search_mac, search_latency
from postpruner.prune.rearrange import rearrange_mask
from postpruner.prune.rescale import rescale_mask
from postpruner.dataset.evaluate.nlp import test_accuracy
from postpruner.utils.schedule import get_pruning_schedule

logger = logging.getLogger(__name__)

def run(model_name, task_name, ckpt_dir, constraint, output_dir=None, gpu=0, metric='mac', mha_lut=None, ffn_lut=None, num_samples=2048, seed=0):
    IS_SQUAD = "squad" in task_name
    IS_LARGE = "large" in model_name
    seq_len = 170 if IS_SQUAD else avg_seq_length(task_name)

    # Create the output directory
    if output_dir is None:
        output_dir = os.path.join(
            "outputs",
            model_name,
            task_name,
            metric,
            str(constraint),
            f"seed_{seed}",
        )
    os.makedirs(output_dir, exist_ok=True)

    # Initiate the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "log.txt")),
        ],
    )

    # Set a GPU and the experiment seed
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    set_seed(seed)
    logger.info(f"Seed number: {seed}")

    # Load the finetuned model and the corresponding tokenizer
    config = AutoConfig.from_pretrained(ckpt_dir)
    model_generator = AutoModelForQuestionAnswering if IS_SQUAD else AutoModelForSequenceClassification
    model = model_generator.from_pretrained(ckpt_dir, config=config)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        use_auth_token=None,
    )

    # Load the training dataset
    if IS_SQUAD:
        training_dataset = squad_dataset(
            task_name,
            tokenizer,
            training=True,
            max_seq_len=384,
            pad_to_max=False,
        )
    else:
        training_dataset = glue_dataset(
            task_name,
            tokenizer,
            training=True,
            max_seq_len=max_seq_length(task_name),
            pad_to_max=False,
        )

    # Sample the examples to be used for search
    collate_fn = DataCollatorWithPadding(tokenizer)
    sample_dataset = Subset(
        training_dataset,
        np.random.choice(len(training_dataset), num_samples).tolist(),
    )
    sample_batch_size = int((12 if IS_SQUAD else 32) * (0.5 if IS_LARGE else 1))
    sample_dataloader = DataLoader(
        sample_dataset,
        batch_size=sample_batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    # Prepare the model
    model = model.cuda()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    full_head_mask = torch.ones(config.num_hidden_layers, config.num_attention_heads).cuda()
    full_neuron_mask = torch.ones(config.num_hidden_layers, config.intermediate_size).cuda()
    
    head_mask_path = os.path.join(output_dir, "head_mask.pt")
    neuron_mask_path = os.path.join(output_dir, "neuron_mask.pt")

    if not os.path.exists(head_mask_path) or not os.path.exists(neuron_mask_path):
        print("No head_mask or neuron_mask exists, pruning")
        start = time.time()
        # Search the optimal mask
        head_grads, neuron_grads = collect_mask_grads(
            model,
            full_head_mask,
            full_neuron_mask,
            sample_dataloader,
        )
        teacher_constraint = get_pruning_schedule(target=constraint, num_iter=2)[0]
        if metric == "mac":
            teacher_head_mask, teacher_neuron_mask = search_mac(
                config,
                head_grads,
                neuron_grads,
                seq_len,
                teacher_constraint,
            )
            head_mask, neuron_mask = search_mac(
                config,
                head_grads,
                neuron_grads,
                seq_len,
                constraint,
            )
            pruned_mac, orig_mac = compute_mask_mac(head_mask, neuron_mask, seq_len, config.hidden_size)
            logger.info(f"Pruned Model MAC: {pruned_mac / orig_mac * 100.0:.2f} %")
        elif metric == "latency":
            mha_lut = torch.load(mha_lut)
            ffn_lut = torch.load(ffn_lut)
            teacher_head_mask, teacher_neuron_mask = search_latency(
                config,
                head_grads,
                neuron_grads,
                teacher_constraint,
                mha_lut,
                ffn_lut,
            )
            head_mask, neuron_mask = search_latency(
                config,
                head_grads,
                neuron_grads,
                constraint,
                mha_lut,
                ffn_lut,
            )
            pruned_latency = estimate_latency(mha_lut, ffn_lut, head_mask, neuron_mask)
            logger.info(f"Pruned Model Latency: {pruned_latency:.2f} ms")

        # Rearrange the mask
        head_mask = rearrange_mask(head_mask, head_grads)
        neuron_mask = rearrange_mask(neuron_mask, neuron_grads)

        # Rescale the mask by solving a least squares problem
        head_mask, neuron_mask = rescale_mask(
            model,
            config,
            teacher_head_mask,
            teacher_neuron_mask,
            head_mask,
            neuron_mask,
            sample_dataloader,
            classification_task=not IS_SQUAD,
        )

        # Print the pruning time
        end = time.time()
        logger.info(f"{task_name} Pruning time (s): {end - start}")
        issaving = True
    else:
        print("head_mask and neuron_mask exists, skip pruning")
        head_mask = torch.load(head_mask_path)
        neuron_mask = torch.load(neuron_mask_path)
        issaving = False

    # Evaluate the accuracy
    test_acc = test_accuracy(model, head_mask, neuron_mask, tokenizer, task_name)
    logger.info(f"{task_name} Test accuracy: {test_acc:.4f}")
    
    # Save the masks
    if issaving:
        torch.save(head_mask, os.path.join(output_dir, "head_mask.pt"))
        torch.save(neuron_mask, os.path.join(output_dir, "neuron_mask.pt"))
