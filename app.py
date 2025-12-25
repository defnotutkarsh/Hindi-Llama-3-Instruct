import gradio as gr
from unsloth import FastLanguageModel
import torch

# --- CONFIGURATION ---
# This points to your saved model on Hugging Face
MODEL_NAME = "defnotutkarsh/hindi-llama-3-slang-v2"

print(f"⏳ Downloading model: {MODEL_NAME}...")

# 1. Load the Model & Tokenizer
# We load directly from Hugging Face. 
# NOTICE: No 'token' argument is used here!
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)

# 2. Prepare Model for Chatting
FastLanguageModel.for_inference(model)

# 3. The Chat Function
def chat_with_slang_ai(user_input):
    # Format the prompt exactly like the training data (Alpaca format)
    prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""".format(user_input, "", "")

    # Generate Response
    inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens = 256, use_cache = True)
    
    # Decode and clean up
    response = tokenizer.batch_decode(outputs)[0]
    final_answer = response.split("### Response:")[-1].replace("<|end_of_text|>", "").strip()
    return final_answer

# 4. The User Interface (UI)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🇮🇳 Hindi Slang AI (V2)")
    gr.Markdown("This AI has been fine-tuned for 300 steps to speak like a real 'Bhai'. Ask it anything!")
    
    with gr.Row():
        user_box = gr.Textbox(label="Apna Sawaal Pucho", placeholder="Bhai, coding start kaise karu?")
    
    output_box = gr.Textbox(label="AI Ka Jawaab")
    btn = gr.Button("Submit / Bhejo", variant="primary")
    
    btn.click(chat_with_slang_ai, inputs=user_box, outputs=output_box)

# 5. Launch
if __name__ == "__main__":
    demo.launch(share=True)