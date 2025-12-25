# 🇮🇳 Hindi-Llama-3-Instruct (Slang/Hinglish)

A fine-tuned version of **Llama-3 8B** optimized for casual **Hindi/Hinglish conversations**. Unlike standard models that speak formal "textbook Hindi," this model understands slang (e.g., "Bhai," "Scene," "Jugaad," "Tension mat le").

## 🌟 Features
- **Slang Fluent:** Trained on 1,000+ synthetic conversations to speak like a real Indian college student/friend.
- **Instruct Tuned:** Follows instructions while maintaining a casual persona.
- **Efficient:** Fine-tuned using **LoRA (Low-Rank Adaptation)** and **Unsloth** for 2x faster training.

## 🔗 Project Links
- **Model Weights (Hugging Face):** [https://huggingface.co/defnotutkarsh/hindi-llama-3-slang-v2](https://huggingface.co/defnotutkarsh/hindi-llama-3-slang-v2)
- **Dataset:** Custom synthetic dataset generated via Llama-3-70B.

## 🛠️ Tech Stack
- **Model:** Meta Llama 3 (8B)
- **Library:** `Unsloth` (for faster QLoRA fine-tuning) + `Hugging Face Transformers`
- **Infrastructure:** Google Colab (T4 GPU)
- **Interface:** Gradio

## 📂 Repository Structure
- `training_pipeline.ipynb`: The full notebook used to train the model (Data Prep -> Training -> Saving).
- `generator.py`: Script used to generate synthetic Hindi datasets.
- `app.py`: A Gradio web interface to chat with the model.
- `requirements.txt`: List of dependencies to run the project.

## 🚀 How to Run Locally

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/defnotutkarsh/Hindi-Llama-3-Instruct.git](https://github.com/defnotutkarsh/Hindi-Llama-3-Instruct.git)
   cd Hindi-Llama-3-Instruct
