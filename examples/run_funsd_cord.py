#!/usr/bin/env python
# coding=utf-8
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from seqeval.metrics import classification_report

import numpy as np
from datasets import ClassLabel, load_dataset
import evaluate

import transformers
import torch
from layoutlmft.data import DataCollatorForKeyValueExtraction
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

logger = logging.getLogger(__name__)
from layoutlmft.data.image_utils import RandomResizedCropAndInterpolationWithTwoPic, pil_loader, Compose

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    use_hierarchical_position_encoding: bool = field(
        default=False,
        metadata={"help": "Enable Hierarchical Position Encoding (HPE)"}
    )
    max_line_position: int = field(
        default=50,
        metadata={"help": "Maximum number of lines in a document for HPE"}
    )
    max_block_position: int = field(
        default=20,
        metadata={"help": "Maximum number of blocks in a document for HPE"}
    )
    use_column_encoding: bool = field(
        default=False,
        metadata={"help": "Enable Column Position Encoding"}
    )
    max_column_position: int = field(
        default=10,
        metadata={"help": "Maximum number of columns in a document"}
    )
    use_intra_line_boundary: bool = field(
        default=False,
        metadata={"help": "Enable Intra-Line Boundary Transition Parsing"}
    )
    lambda_bound_init: float = field(
        default=0.1,
        metadata={"help": "Initial weight for boundary loss"}
    )
    use_semantic_geometry_disentangle: bool = field(
        default=False,
        metadata={"help": "Enable Semantic-Geometry Disentanglement"}
    )
    lambda_geo_init: float = field(
        default=0.1,
        metadata={"help": "Initial weight for geometry loss"}
    )
    lambda_orth_init: float = field(
        default=0.1,
        metadata={"help": "Initial weight for orthogonality loss"}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    segment_level_layout: bool = field(default=True)
    visual_embed: bool = field(default=True)
    use_segment_head: bool = field(
        default=False,
        metadata={
            "help": "Use LayoutLMv3ForSegmentTokenClassification (segment-level pooling + "
            "inter-segment context head) instead of the vanilla per-token classification head."
        },
    )
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})
    geo_y_threshold: float = field(
        default=10.0, metadata={"help": "Threshold for line clustering on Y axis"}
    )
    geo_x_threshold: float = field(
        default=50.0, metadata={"help": "Threshold for column/block clustering on X axis"}
    )

def main():
    # See all possible arguments in layoutlmft/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    if data_args.dataset_name == 'funsd':
        # datasets = load_dataset("nielsr/funsd")
        import layoutlmft.data.funsd
        datasets = load_dataset(os.path.abspath(layoutlmft.data.funsd.__file__), cache_dir=model_args.cache_dir)
    elif data_args.dataset_name == 'cord':
        import layoutlmft.data.cord
        datasets = load_dataset(os.path.abspath(layoutlmft.data.cord.__file__), cache_dir=model_args.cache_dir)
    else:
        raise NotImplementedError()

    if training_args.do_train:
        column_names = datasets["train"].column_names
        features = datasets["train"].features
    else:
        column_names = datasets["test"].column_names
        features = datasets["test"].features

    text_column_name = "words" if "words" in column_names else "tokens"

    label_column_name = (
        f"{data_args.task_name}_tags" if f"{data_args.task_name}_tags" in column_names else column_names[1]
    )

    remove_columns = column_names

    # In the event the labels are not a `Sequence[ClassLabel]`, we will need to go through the dataset to get the
    # unique labels.
    def get_label_list(labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list

    if isinstance(features[label_column_name].feature, ClassLabel):
        label_list = features[label_column_name].feature.names
        label_to_id = {i: i for i in range(len(label_list))}
    else:
        label_list = get_label_list(datasets["train"][label_column_name])
        label_to_id = {l: i for i, l in enumerate(label_list)}
        
    num_labels = len(label_list)

    # THÊM KHỐI NÀY ĐỂ BẢO VỆ LOGIC BOUNDARY LOSS:
    if getattr(model_args, "use_intra_line_boundary", False):
        try:
            assert label_list[0] == "O", "Nhãn đầu tiên phải là 'O'"
            for i in range(1, len(label_list), 2):
                if i + 1 < len(label_list):
                    assert label_list[i].startswith("B-"), f"Nhãn {i} phải là B-, nhưng là {label_list[i]}"
                    assert label_list[i+1].startswith("I-"), f"Nhãn {i+1} phải là I-, nhưng là {label_list[i+1]}"
                    assert label_list[i][2:] == label_list[i+1][2:], "B- và I- không khớp loại entity"
            logger.info("✅ Label list pass B-/I- parity check cho Boundary Loss.")
        except AssertionError as e:
            raise ValueError(f"Label list không đúng chuẩn B-/I- luân phiên. Cần sửa logic boundary loss. Chi tiết: {e}")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        input_size=data_args.input_size,
        use_auth_token=True if model_args.use_auth_token else None,
        use_hierarchical_position_encoding=model_args.use_hierarchical_position_encoding,
        max_line_position=model_args.max_line_position,
        max_block_position=model_args.max_block_position,
        use_column_encoding=model_args.use_column_encoding,
        max_column_position=model_args.max_column_position,
        use_intra_line_boundary=model_args.use_intra_line_boundary,
        lambda_bound_init=model_args.lambda_bound_init,
        use_semantic_geometry_disentangle=model_args.use_semantic_geometry_disentangle,
        lambda_geo_init=model_args.lambda_geo_init,
        lambda_orth_init=model_args.lambda_orth_init,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
        cache_dir=model_args.cache_dir,
        use_fast=True,
        add_prefix_space=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if getattr(data_args, "use_segment_head", False):
        # NEW: segment-level pooling + inter-segment context head.
        # See modeling_layoutlmv3_segment.py for the full design rationale.
        from layoutlmft.models.layoutlmv3.modeling_layoutlmv3_segment import (
    LayoutLMv3ForSegmentTokenClassification,
)
        model = LayoutLMv3ForSegmentTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
            "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
            "requirement"
        )

    # Preprocessing the dataset
    # Padding strategy
    padding = "max_length" if data_args.pad_to_max_length else False

    if data_args.visual_embed:
        imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
        mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
        std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
        common_transform = Compose([
            # transforms.ColorJitter(0.4, 0.4, 0.4),
            # transforms.RandomHorizontalFlip(p=0.5),
            RandomResizedCropAndInterpolationWithTwoPic(
                size=data_args.input_size, interpolation=data_args.train_interpolation),
        ])
        import torch
        patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std))
        ])

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples, augmentation=False):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            padding=False,
            truncation=True,
            return_overflowing_tokens=True,
            is_split_into_words=True,
        )

        labels = []
        bboxes = []
        images = []
        seg_ids = []
        line_ids_all = []    # NEW
        block_ids_all = []   # NEW
        column_ids_all = []
        
        # HÀM MỚI (PHASE 1): Clustering 2D có sort trước
        def compute_geometry_ids(bboxes, y_thresh=10.0, x_thresh=50.0):
            n = len(bboxes)
            if n == 0:
                return [], [], []
            
            # Sort toàn bộ box theo (y_center, x_center)
            order = sorted(range(n), key=lambda i: ((bboxes[i][1] + bboxes[i][3]) / 2,
                                                    (bboxes[i][0] + bboxes[i][2]) / 2))
            
            # --- Gom dòng (Line) ---
            line_of_sorted = [0] * n
            cur_line = 0
            prev_y = (bboxes[order[0]][1] + bboxes[order[0]][3]) / 2
            for k in range(1, n):
                y = (bboxes[order[k]][1] + bboxes[order[k]][3]) / 2
                if abs(y - prev_y) > y_thresh:
                    cur_line += 1
                line_of_sorted[k] = cur_line
                prev_y = y

            # --- Gom cột (Column) trong từng dòng ---
            col_of_sorted = [0] * n
            idx_by_line = {}
            for k, ln in enumerate(line_of_sorted):
                idx_by_line.setdefault(ln, []).append(k)
                
            for ln, ks in idx_by_line.items():
                ks_sorted = sorted(ks, key=lambda k: (bboxes[order[k]][0] + bboxes[order[k]][2]) / 2)
                cur_col = 0
                prev_x = None
                for k in ks_sorted:
                    x = (bboxes[order[k]][0] + bboxes[order[k]][2]) / 2
                    if prev_x is not None and abs(x - prev_x) > x_thresh:
                        cur_col += 1
                    col_of_sorted[k] = cur_col
                    prev_x = x

            # --- Gom khối (Block) dựa trên khoảng cách 2D ---
            block_of_sorted = [0] * n
            cur_block = 0
            prev_center = None
            for k in range(n):
                cx = (bboxes[order[k]][0] + bboxes[order[k]][2]) / 2
                cy = (bboxes[order[k]][1] + bboxes[order[k]][3]) / 2
                if prev_center is not None:
                    dx, dy = abs(cx - prev_center[0]), abs(cy - prev_center[1])
                    # Nới lỏng threshold cho block (union các dòng gần nhau)
                    if dx > x_thresh * 3 or dy > y_thresh * 3:
                        cur_block += 1
                block_of_sorted[k] = cur_block
                prev_center = (cx, cy)

            # --- Trả ngược về thứ tự gốc của dataset ---
            line_ids = [0] * n
            block_ids = [0] * n
            col_ids = [0] * n
            for k, orig_i in enumerate(order):
                line_ids[orig_i] = line_of_sorted[k]
                block_ids[orig_i] = block_of_sorted[k]
                col_ids[orig_i] = col_of_sorted[k]
                
            return line_ids, block_ids, col_ids
        
        for batch_index in range(len(tokenized_inputs["input_ids"])):
            word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
            org_batch_index = tokenized_inputs["overflow_to_sample_mapping"][batch_index]

            label = examples[label_column_name][org_batch_index]
            bbox = examples["bboxes"][org_batch_index]

            line_ids_orig, block_ids_orig, column_ids_orig = compute_geometry_ids(
                bbox, 
                y_thresh=data_args.geo_y_threshold, 
                x_thresh=data_args.geo_x_threshold
            )

            # NEW: recover segment boundaries (giữ nguyên code cũ)
            word_seg_id = None
            if getattr(data_args, "use_segment_head", False):
                word_seg_id = []
                seg_counter = -1
                prev_bbox_tuple = None
                for wb in bbox:
                    wb_tuple = tuple(wb)
                    if wb_tuple != prev_bbox_tuple:
                        seg_counter += 1
                        prev_bbox_tuple = wb_tuple
                    word_seg_id.append(seg_counter)

            previous_word_idx = None
            label_ids = []
            bbox_inputs = []
            seg_id_inputs = []
            line_ids_aligned = []    # NEW
            block_ids_aligned = []   # NEW
            column_ids_aligned = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    # Special tokens
                    label_ids.append(-100)
                    bbox_inputs.append([0, 0, 0, 0])
                    if word_seg_id is not None:
                        seg_id_inputs.append(-1)
                    line_ids_aligned.append(-1)     # NEW
                    block_ids_aligned.append(-1)    # NEW
                    column_ids_aligned.append(-1)
                elif word_idx != previous_word_idx:
                    # First token of a word
                    label_ids.append(label_to_id[label[word_idx]])
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                    line_ids_aligned.append(line_ids_orig[word_idx])     # NEW
                    block_ids_aligned.append(block_ids_orig[word_idx])   # NEW
                    column_ids_aligned.append(column_ids_orig[word_idx])
                else:
                    # Subsequent tokens of the same word
                    label_ids.append(label_to_id[label[word_idx]] if data_args.label_all_tokens else -100)
                    bbox_inputs.append(bbox[word_idx])
                    if word_seg_id is not None:
                        seg_id_inputs.append(word_seg_id[word_idx])
                    line_ids_aligned.append(line_ids_orig[word_idx])     # NEW
                    block_ids_aligned.append(block_ids_orig[word_idx])   # NEW
                    column_ids_aligned.append(column_ids_orig[word_idx])
                previous_word_idx = word_idx
                
            labels.append(label_ids)
            bboxes.append(bbox_inputs)
            if word_seg_id is not None:
                seg_ids.append(seg_id_inputs)
            line_ids_all.append(line_ids_aligned)     # NEW
            block_ids_all.append(block_ids_aligned)   # NEW
            column_ids_all.append(column_ids_aligned)

            if data_args.visual_embed:
                ipath = examples["image_path"][org_batch_index]
                img = pil_loader(ipath)
                for_patches, _ = common_transform(img, augmentation=augmentation)
                patch = patch_transform(for_patches)
                images.append(patch)

        tokenized_inputs["labels"] = labels
        tokenized_inputs["bbox"] = bboxes
        tokenized_inputs["line_ids"] = line_ids_all    # NEW
        tokenized_inputs["block_ids"] = block_ids_all  # NEW
        tokenized_inputs["column_ids"] = column_ids_all
        
        if getattr(data_args, "use_segment_head", False):
            tokenized_inputs["seg_id"] = seg_ids
        if data_args.visual_embed:
            tokenized_inputs["images"] = images

        tokenized_inputs.pop("overflow_to_sample_mapping", None)
        tokenized_inputs.pop("offset_mapping", None)

        return tokenized_inputs

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        

    if training_args.do_eval:
        validation_name = "test"
        if validation_name not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets[validation_name]
        if data_args.max_val_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        if data_args.max_test_samples is not None:
            test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            tokenize_and_align_labels,
            batched=True,
            remove_columns=remove_columns,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )
    # ====== KIỂM TRA BATCH DATA ======
    # Tạo data collator và dataloader để kiểm tra
    from torch.utils.data import DataLoader
    temp_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        collate_fn=data_collator,
        shuffle=False
    )
    
    # Lấy 1 batch
    batch = next(iter(temp_dataloader))
    
    # Kiểm tra các keys trong batch
    print("=" * 50)
    print("KEYS IN BATCH:", batch.keys())
    print("=" * 50)
    
    # Kiểm tra line_ids và block_ids có tồn tại không
    if "line_ids" in batch:
        print(f"✅ line_ids shape: {batch['line_ids'].shape}")
        print(f"   line_ids sample: {batch['line_ids'][0][:10]}")  # 10 token đầu
    else:
        print("❌ line_ids NOT FOUND in batch!")
    
    if "block_ids" in batch:
        print(f"✅ block_ids shape: {batch['block_ids'].shape}")
        print(f"   block_ids sample: {batch['block_ids'][0][:10]}")
    else:
        print("❌ block_ids NOT FOUND in batch!")
    
    # Kiểm tra seg_id có bị xóa không
    if "seg_id" in batch:
        print(f"✅ seg_id shape: {batch['seg_id'].shape}")
    else:
        print("⚠️ seg_id NOT FOUND (có thể bị xóa trong data_collator)")
    
    print("=" * 50)
    # ====== KẾT THÚC KIỂM TRA ======

    # Metrics
    metric = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)

        # Tạo bảng thống kê lỗi (Precision, Recall, F1 cho từng nhãn)
        report = classification_report(true_labels, true_predictions)
        
        # Chỉ lưu vào file trong thư mục output (KHÔNG in ra màn hình log)
        report_file_path = os.path.join(training_args.output_dir, "eval_classification_report.txt")
        with open(report_file_path, "w", encoding="utf-8") as f:
            f.write("="*50 + "\n")
            f.write("📊 THỐNG KÊ LỖI / CHI TIẾT TỪNG NHÃN (EVAL MỚI NHẤT)\n")
            f.write("="*50 + "\n")
            f.write(report + "\n")

        if data_args.return_entity_level_metrics:
            # Unpack nested dictionaries
            final_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    for n, v in value.items():
                        final_results[f"{key}_{n}"] = v
                else:
                    final_results[key] = value
            return final_results
        else:
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }
    import torch
    # Định nghĩa Trainer tùy chỉnh để tách biệt Learning Rate
    class CustomTrainer(Trainer):
        def create_optimizer(self):
            if self.optimizer is None:
                # Nhóm 1: Các tham số thuộc backbone LayoutLMv3
                backbone_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" in n and p.requires_grad]
                # Nhóm 2: Các tham số mới (segment_context, classifier, is_first_token_embedding, gate)
                new_params = [p for n, p in self.model.named_parameters() if "layoutlmv3" not in n and p.requires_grad]

                optimizer_grouped_parameters = [
                    {"params": backbone_params, "lr": self.args.learning_rate}, # Dùng LR từ tham số truyền vào (VD: 1e-5)
                    {"params": new_params, "lr":5e-4} # Ép cứng LR lớn hơn cho module mới
                ]
                
                self.optimizer = torch.optim.AdamW(
                    optimizer_grouped_parameters, 
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            return self.optimizer

    # Khởi tạo Trainer bằng CustomTrainer vừa tạo thay vì Trainer mặc định
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # Initialize our Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset if training_args.do_train else None,
    #     eval_dataset=eval_dataset if training_args.do_eval else None,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator,
    #     compute_metrics=compute_metrics,
    # )

    # Training
    if training_args.do_train:
        checkpoint = last_checkpoint if last_checkpoint else None
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_val_samples = data_args.max_val_samples if data_args.max_val_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_val_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predictions, labels, metrics = trainer.predict(test_dataset)
        
        # Đổi tên biến thành pred_argmax để giữ nguyên predictions gốc cho việc trích xuất
        pred_argmax = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens) cho việc tính log gốc
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(pred_argmax, labels)
        ]

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        # =====================================================================
        # THÊM MỚI: LIỆT KÊ LỖI DỰ ĐOÁN VÀ XUẤT RA CSV
        # =====================================================================
        import csv
        error_output_file = os.path.join(training_args.output_dir, "detailed_prediction_errors.csv")
        
        # Chỉ ghi file bằng tiến trình chính (tránh lỗi xung đột khi chạy multi-GPU)
        if trainer.is_world_process_zero():
            with open(error_output_file, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                # Ghi header của file CSV
                writer.writerow(["Sample_Index", "Token_Index", "Text", "Bounding_Box", "True_Label", "Predicted_Label", "Error_Type"])
                
                # Duyệt qua các mẫu trong tập test
                for i, (pred, label) in enumerate(zip(pred_argmax, labels)):
                    # Trích xuất dữ liệu thô từ test_dataset để đối chiếu
                    input_ids = test_dataset[i]["input_ids"]
                    bboxes = test_dataset[i]["bbox"]
                    # Chuyển đổi input_ids thành các token chữ
                    tokens = tokenizer.convert_ids_to_tokens(input_ids)
                    
                    for j, (p, l) in enumerate(zip(pred, label)):
                        if l != -100:  # Bỏ qua padding, [CLS], [SEP]
                            true_l = label_list[l]
                            pred_l = label_list[p]
                            
                            # Nếu dự đoán sai thì tiến hành ghi lỗi
                            if true_l != pred_l:
                                # Làm sạch token (loại bỏ ký tự đặc biệt của tokenizer như 'Ġ' hoặc ' ')
                                token_text = tokens[j].replace("Ġ", "").replace(" ", "")
                                bbox = bboxes[j]
                                
                                # Phân loại nguyên nhân lỗi
                                if true_l == "O" and pred_l != "O":
                                    err_type = "False Positive (Nhận diện thừa)"
                                elif true_l != "O" and pred_l == "O":
                                    err_type = "False Negative (Bỏ sót)"
                                else:
                                    err_type = "Misclassification (Nhầm nhãn)"
                                    
                                writer.writerow([i, j, token_text, bbox, true_l, pred_l, err_type])
            
            logger.info(f"Đã lưu danh sách lỗi dự đoán chi tiết tại: {error_output_file}")
        # =====================================================================

        # Save predictions (Giữ nguyên đoạn code ghi file .txt cũ)
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                for prediction in true_predictions:
                    writer.write(" ".join(prediction) + "\n")


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()