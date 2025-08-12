import sys
import os

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GraniteMoeHybridForCausalLM
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

# MODEL_ID = "ibm-granite/granite-4.0-tiny-preview"
model_path = sys.argv[1]
store_path = sys.argv[2]
print(f"Quantizing {model_path} using FP8_DYNAMIC...")

model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto",
)
# model = GraniteMoeHybridForCausalLM.from_pretrained(
#     model_path, device_map="auto", torch_dtype="auto",
# )
# print(model)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Configure the simple PTQ quantization
recipe = QuantizationModifier(
  targets="Linear", scheme="FP8_DYNAMIC", 
  ignore=[
        "re:.*lm_head",
        # "re:.*block_sparse_moe",
        "re:.*block_sparse_moe.router",
  ]
  )

# Apply the quantization algorithm.
oneshot(model=model, recipe=recipe)
#, output_dir=store_path)

print(f"...done. Saving to {store_path}...")
# # SAVE_DIR = MODEL_ID.split("/")[1] + "-FP8-Dynamic"
model.save_pretrained(store_path, save_compressed=True)
tokenizer.save_pretrained(store_path)

print("...done.")
