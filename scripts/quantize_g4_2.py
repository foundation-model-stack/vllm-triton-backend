import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"
model_path = sys.argv[1]
store_path = sys.argv[2]
print(f"Quantizing {model_path} using FP8...")


from datasets import load_dataset
from packaging.version import Version
from transformers import AutoModelForCausalLM, AutoTokenizer, __version__

from llmcompressor.utils import dispatch_for_generation

# NOTE: transformers 4.49.0 has an attribute error with DeepSeek.
# Please consider either downgrading your transformers version to a
# previous version or upgrading to a version where this bug is fixed

# select a Mixture of Experts model for quantization
MODEL_ID = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, torch_dtype="auto", trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Select calibration dataset.
# its recommended to use more calibration samples for MoE models so each expert is hit
DATASET_ID = "HuggingFaceH4/ultrachat_200k"
DATASET_SPLIT = "train_sft"
NUM_CALIBRATION_SAMPLES = 2048
MAX_SEQUENCE_LENGTH = 2048


# Load dataset and preprocess.
ds = load_dataset(DATASET_ID, split=f"{DATASET_SPLIT}[:{NUM_CALIBRATION_SAMPLES}]")
ds = ds.shuffle(seed=42)


def preprocess(example):
    return {
        "text": tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
        )
    }


ds = ds.map(preprocess)


# Tokenize inputs.
def tokenize(sample):
    return tokenizer(
        sample["text"],
        padding=False,
        max_length=MAX_SEQUENCE_LENGTH,
        truncation=True,
        add_special_tokens=False,
    )


ds = ds.map(tokenize, remove_columns=ds.column_names)



model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8", 
  ignore=[
        "re:.*lm_head",
        "re:.*self_attn",
        "re:.*router",
        # "re:.*block_sparse_moe.gate",
        "re:.*moe*",
  ]
  )

# Apply the quantization algorithm.
# oneshot(model=model, recipe=recipe)

oneshot(
    model=model,
    dataset=ds,
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    trust_remote_code_model=True,
)


print(f"...done. Saving to {store_path}...")
# Save the model: granite-4.0-tiny-preview-FP8-Dynamic
# SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(store_path)
tokenizer.save_pretrained(store_path)

print("...done.")
