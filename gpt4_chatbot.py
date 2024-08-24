import os
import gradio as gr
from openai import OpenAI, OpenAIError, RateLimitError
import time
import json

# Use an environment variable for the API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=api_key)

# Model information including pricing (approximate, may need updating)
MODEL_INFO = {
    "gpt-4": {"price_per_1k_tokens": 0.03, "quality": 10},
    "gpt-4-turbo-preview": {"price_per_1k_tokens": 0.01, "quality": 9},
    "gpt-3.5-turbo": {"price_per_1k_tokens": 0.0015, "quality": 7},
    "gpt-3.5-turbo-16k": {"price_per_1k_tokens": 0.003, "quality": 7},
    "dall-e-3": {"price": 0.04, "quality": 10},  # Price per image (1024x1024)
    "text-embedding-3-small": {"price_per_1k_tokens": 0.00002, "quality": 8},
    "text-embedding-3-large": {"price_per_1k_tokens": 0.00013, "quality": 10},
}

def calculate_cost(model, tokens):
    if model == "dall-e-3":
        return MODEL_INFO[model]["price"]  # DALL-E 3 has a fixed price per image
    return (tokens / 1000) * MODEL_INFO[model]["price_per_1k_tokens"]

def select_model(message_length, desired_quality, max_cost):
    best_model = None
    best_score = float('-inf')
    
    for model, info in MODEL_INFO.items():
        if "price_per_1k_tokens" not in info:
            continue
        estimated_cost = calculate_cost(model, message_length)
        if estimated_cost <= max_cost:
            quality_score = info["quality"]
            score = quality_score - abs(quality_score - desired_quality)
            if score > best_score:
                best_score = score
                best_model = model
    
    return best_model or "gpt-3.5-turbo"

def generate_image(prompt):
    print(f"Attempting to generate image with prompt: {prompt}")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        print(f"Image generated successfully. URL: {response.data[0].url}")
        return response.data[0].url
    except Exception as e:
        print(f"Error generating image: {str(e)}")
        return f"Error generating image: {str(e)}"

def get_embedding(text, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            model=model,
            input=text
        )
        return response.data[0].embedding[:10]  # Return only first 10 elements for brevity
    except Exception as e:
        return f"Error generating embedding: {str(e)}"

def snow_ai_chatbot(message, history, desired_quality, max_cost, temperature, max_tokens, selected_model):
    print("Function called with message:", message)
    print("Selected model:", selected_model)
    
    # Explicitly check for image generation request
    if any(phrase in str(message).lower() for phrase in ["generate an image", "create an image", "dall-e"]):
        print("Image generation request detected")
        image_url = generate_image(message)
        ai_message = f"I've generated an image based on your request. Here's the URL: {image_url}"
        history.append((str(message), ai_message))
        return history

    history_openai_format = [
        {"role": "system", "content": "You are Snow AI, a helpful and friendly AI assistant."}
    ]
    for human, assistant in history:
        if human:
            history_openai_format.append({"role": "user", "content": str(human)})
        if assistant:
            history_openai_format.append({"role": "assistant", "content": str(assistant)})
    history_openai_format.append({"role": "user", "content": str(message)})
    
    print("History formatted for OpenAI:", history_openai_format)
    
    try:
        if "create an embedding" in str(message).lower():
            print("Creating embedding")
            embedding = get_embedding(message)
            ai_message = f"I've created an embedding for your text. Here are the first 10 elements: {embedding[:10]}"
        else:
            print("Using chat model")
            
            response = client.chat.completions.create(
                model=selected_model,
                messages=history_openai_format,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            ai_message = response.choices[0].message.content
            cost = calculate_cost(selected_model, response.usage.total_tokens)
            ai_message = f"[Model: {selected_model}, Cost: ${cost:.4f}]\n{ai_message}"

        print("AI message:", ai_message)

        history.append((str(message), ai_message))
        return history

    except Exception as e:
        print("Error occurred:", str(e))
        error_message = f"An error occurred: {str(e)}"
        history.append((str(message), error_message))
        return history

with gr.Blocks() as iface:
    gr.Markdown("# Snow AI - Your Friendly AI Assistant")
    gr.Markdown("Chat with Snow AI! You can type your message. Ask for image generation or text embeddings within your chat.")

    chatbot = gr.Chatbot(label="Snow AI Chat")
    msg = gr.Textbox(placeholder="Chat with Snow AI...", label="Your message")

    with gr.Row():
        model_dropdown = gr.Dropdown(
            choices=["gpt-4", "gpt-4-turbo-preview", "gpt-3.5-turbo", "gpt-3.5-turbo-16k", "dall-e-3"],
            value="gpt-3.5-turbo",
            label="Select Model"
        )
        submit_btn = gr.Button("Submit")

    with gr.Accordion("Advanced Settings", open=False):
        desired_quality = gr.Slider(1, 10, value=7, step=1, label="Desired Quality (1-10)")
        max_cost = gr.Slider(0.01, 0.1, value=0.05, label="Max Cost per Response ($)")
        temperature = gr.Slider(0, 1, value=0.7, label="Temperature")
        max_tokens = gr.Slider(50, 500, value=150, step=10, label="Max Tokens")

    def user_input(user_message, history):
        return "", history + [[user_message, None]]

    msg.submit(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        snow_ai_chatbot,
        [msg, chatbot, desired_quality, max_cost, temperature, max_tokens, model_dropdown],
        [chatbot]
    )

    submit_btn.click(user_input, [msg, chatbot], [msg, chatbot], queue=False).then(
        snow_ai_chatbot,
        [msg, chatbot, desired_quality, max_cost, temperature, max_tokens, model_dropdown],
        [chatbot]
    )

if __name__ == "__main__":
    iface.launch(debug=True)