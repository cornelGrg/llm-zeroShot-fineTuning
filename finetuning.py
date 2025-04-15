from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

if __name__ == "__main__":
    model_id = "google/gemma-3-1b-it"

    # Load model on MPS (Apple Silicon GPU) with float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to("mps").eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}],
            },
        ],
    ]

    # No bfloat16, just tokenize normally
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Move tensors to MPS
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=64)

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(decoded[0])
