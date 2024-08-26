import os
import gradio as gr
from openai import OpenAI
import fal_client
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import requests
import base64
import tiktoken

# Load environment variables
load_dotenv()

# Use environment variables for API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
fal_key = os.getenv("FAL_KEY")

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")
if not fal_key:
    raise ValueError("FAL_KEY not found in environment variables")
print(f"FAL_KEY found: {'*' * len(fal_key)}")

client = OpenAI(api_key=openai_api_key)

def generate_image_fal(prompt, fal_model):
    print(f"Attempting to generate image with FAL AI. Model: {fal_model}, Prompt: {prompt}")
    
    # Set max_steps based on the model
    if fal_model == "fal-ai/flux/schnell":
        max_steps = 8
    elif fal_model == "fal-ai/stable-diffusion-v3-medium":
        max_steps = 30
    elif fal_model == "fal-ai/flux-realism":
        max_steps = 40
    else:  # fal-ai/flux (dev)
        max_steps = 35
    
    try:
        handler = fal_client.submit(
            fal_model,
            arguments={
                "prompt": prompt,
                "image_size": "landscape_16_9",
                "num_inference_steps": max_steps,
                "guidance_scale": 7.5,
                "enable_safety_checker": False
            },
            
        )
        result = handler.get()
        image_url = result['images'][0]['url']
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image with FAL AI: {str(e)}")
        print(f"Full error details: {e.response.text if hasattr(e, 'response') else 'No additional details'}")
        return None

def generate_image_openai(prompt):
    print(f"Attempting to generate image with DALL-E. Prompt: {prompt}")
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1792x1024",
            quality="standard",
            n=1,
        )
        image_url = response.data[0].url
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content))
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error generating image with DALL-E: {str(e)}")
        return None

def text_chat(message, history, model):
    print(f"text_chat called with message: '{message}', model: '{model}'")
    history = history or []
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for human, ai in history:
        messages.append({"role": "user", "content": human})
        if ai:
            messages.append({"role": "assistant", "content": ai})
    messages.append({"role": "user", "content": message})
    
    try:
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4-turbo"]:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            ai_message = response.choices[0].message.content
        elif model in ["gpt-4o-mini", "chatgpt-4o-latest"]:
            # Assuming these are FAL AI models
            response = fal_client.submit(
                "fal-ai/gpt4all",
                arguments={
                    "model": model,
                    "messages": messages,
                    "max_tokens": 1000,
                    "temperature": 0.7,
                }
            )
            result = response.get()
            ai_message = result['choices'][0]['message']['content']
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        history.append((message, ai_message))
        return history
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        history.append((message, error_message))
        return history

def generate_image(prompt, model):
    print(f"generate_image called with prompt: '{prompt}', model: '{model}'")
    if model.startswith("fal-"):
        if model == "fal-flux-dev1":
            fal_model = "fal-ai/flux"
        elif model == "fal-flux-schnell":
            fal_model = "fal-ai/flux/schnell"
        elif model == "fal-sd-v3-medium":
            fal_model = "fal-ai/stable-diffusion-v3-medium"
        elif model == "fal-flux-realism":
            fal_model = "fal-ai/flux-realism"
        image_data = generate_image_fal(prompt, fal_model)
    else:  # DALL-E
        image_data = generate_image_openai(prompt)
    
    if image_data is None:
        return None
    
    # Convert base64 string to PIL Image
    image_data = image_data.split(",")[1]  # Remove the "data:image/png;base64," part
    image = Image.open(BytesIO(base64.b64decode(image_data)))
    
    return image

css = """
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

body {
    background-color: #000000;
    color: #00ffff;
    font-family: 'Share Tech Mono', monospace;
    margin: 0;
    padding: 0;
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

#app-container {
    display: flex;
    flex-direction: row;
    flex-grow: 1;
    background-color: #000000;
    padding: 10px;
}

#image-column, #chat-column {
    border: 1px solid #00ffff;
    box-shadow: 0 0 10px #00ffff;
    padding: 10px;
    background-color: #0a0a0a;
    display: flex;
    flex-direction: column;
}

#image-column {
    flex: 9;
}

#chat-column {
    flex: 1;
    max-width: 300px;
    display: flex;
    flex-direction: column;
    height: 112vh;
}

#chatbot {
    flex-grow: 2;
    overflow-y: auto;
    font-size: 0.8em;
    background-color: #0f0f0f;
    border: 1px solid #00ffff;
    padding: 10px;
    margin-bottom: 10px;
    position: relative;
}



#chat-model-selector {
    background-color: transparent;
    border: none;
    color: #00ffff;
    font-size: 0.7em;
    cursor: pointer;
    padding: 0;
}

#chat-model-selector::after {
    content: '▼';
    margin-left: 5px;
}

#msg {
    margin-top: 10px;
    width: 100%;
}

#image-output {
    flex-grow: 1;
    width: 100%;
    height: 75vh;
    object-fit: contain;
    border: 1px solid #00ffff;
}

#image-prompt-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    margin-top: 10px;
}

#image-prompt-row {
    display: flex;
    gap: 10px;
}

#image-prompt, #msg {
    flex-grow: 1;
    background-color: #0f0f0f;
    color: #00ffff;
    border: 1px solid #00ffff;
}

#generate-btn, #clear {
    min-width: 120px;
    height: 36px;
    background-color: #0f0f0f;
    color: #00ffff;
    border: 1px solid #00ffff;
    cursor: pointer;
    transition: all 0.3s ease;
}

#generate-btn:hover, #clear:hover {
    background-color: #00ffff;
    color: #000000;
}

#image-model {
    width: 100%;
    background-color: #0f0f0f;
    color: #00ffff;
    border: 1px solid #00ffff;
}

#chat-model-button {
    position: absolute;
    top: 5px;
    right: 5px;
    background: none;
    border: none;
    color: #00ffff;
    font-size: 0.7em;
    cursor: pointer;
    z-index: 1000;
}

#chat-model-dropdown {
    display: none;
    position: absolute;
    top: 25px;
    right: 5px;
    background-color: #0a0a0a;
    border: 1px solid #00ffff;
    z-index: 1001;
}

#chat-model-dropdown button {
    display: block;
    width: 100%;
    padding: 5px 10px;
    background: none;
    border: none;
    color: #00ffff;
    text-align: left;
    cursor: pointer;
}

#chat-model-dropdown button:hover {
    background-color: #00ffff;
    color: #0a0a0a;
}

.title {
    text-align: center;
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
}

.label {
    color: #00ffff;
}

footer {
    background-color: #0a0a0a;
    color: #00ffff;
    text-align: center;
    padding: 10px;
    border-top: 1px solid #00ffff;
    box-shadow: 0 -5px 10px rgba(0, 255, 255, 0.1);
}

#button-row {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
}
"""

with gr.Blocks(css=css, theme=gr.themes.Base()) as iface:
    with gr.Row(elem_id="app-container"):
        with gr.Column(elem_id="image-column"):
            gr.Markdown("# Snow AI Assistant", elem_classes="title")
            gr.Markdown("## Image Generation", elem_classes="title")
            image_output = gr.Image(label="Generated Image", elem_id="image-output")
            with gr.Column(elem_id="image-prompt-container"):
                with gr.Row(elem_id="image-prompt-row"):
                    image_prompt = gr.Textbox(placeholder="Enter image prompt here...", label="Image Prompt", elem_id="image-prompt")
                    image_model = gr.Dropdown(
                        choices=["dall-e-3", "fal-flux-dev1", "fal-flux-schnell", "fal-sd-v3-medium", "fal-flux-realism"],
                        label="Model",
                        value="dall-e-3",
                        elem_id="image-model"
                    )

        with gr.Column(elem_id="chat-column"):
            gr.Markdown("## Chat", elem_classes="title")
            with gr.Column(elem_id="chatbot"):
                chatbot = gr.Chatbot()
                with gr.Column(elem_id="chat-model-container"):
                    chat_model = gr.Dropdown(
                        choices=["gpt-3.5-turbo", "gpt-3.5-turbo-16k", "gpt-4o-mini", "chatgpt-4o-latest", "gpt-4-turbo"],
                        value="gpt-3.5-turbo",
                        label="",
                        elem_id="chat-model-selector"
                    )
            msg = gr.Textbox(placeholder="Type your message here...", label="Your message", elem_id="msg")
    
    with gr.Row(elem_id="button-row"):
        image_button = gr.Button("Generate Image", elem_id="generate-btn")
        clear = gr.Button("Clear", elem_id="clear")

    footer = gr.HTML("<footer>© 2024 Snow AI Assistant. All rights reserved.</footer>")

    def on_image_prompt_submit(prompt, model):
        return generate_image(prompt, model)

    image_button.click(
        on_image_prompt_submit,
        inputs=[image_prompt, image_model],
        outputs=[image_output]
    )
    image_prompt.submit(
        on_image_prompt_submit,
        inputs=[image_prompt, image_model],
        outputs=[image_output]
    )

    def chat_and_clear(message, history, model):
        result = text_chat(message, history, model)
        return result, ""

    msg.submit(chat_and_clear, [msg, chatbot, chat_model], [chatbot, msg])
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    iface.launch(debug=True)