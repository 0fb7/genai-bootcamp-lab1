import gradio as gr
import torch
from PIL import Image
from transformers import (BlipProcessor, BlipForConditionalGeneration,
                          AutoTokenizer, AutoModelForCausalLM)

APP_TITLE   = "Yazeed — Technical Image Analyzer"
ANALYSIS_STYLE = "bulleted technical report"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

blip_id = "Salesforce/blip-image-captioning-base"
blip_processor = BlipProcessor.from_pretrained(blip_id)
blip_model = BlipForConditionalGeneration.from_pretrained(blip_id, dtype=dtype).to(device)

llm_id = "Qwen/Qwen2.5-1.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(llm_id)
llm = AutoModelForCausalLM.from_pretrained(llm_id, dtype=dtype).to(device)

def caption_image(img):
    inputs = blip_processor(img, return_tensors="pt").to(device, dtype)
    out = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out[0], skip_special_tokens=True)

def analyze_technical_image(caption):
    prompt = (f"Perform a detailed engineering analysis based on the following image description: '{caption}'. "
              f"Provide the output as a {ANALYSIS_STYLE}. "
              "Break down the analysis into the following structure: \n1. Identified Components\n2. Potential Circuit Function\n3. Logical Operational Sequence.")
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=False).to(device)
    
    # Lower temperature for deterministic and logical output
    out = llm.generate(inputs, max_new_tokens=250, do_sample=True, temperature=0.3,
                       top_p=0.95, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

def app(img):
    if img is None:
        return "Please upload an image.", ""
    cap = caption_image(img.convert("RGB"))
    return cap, analyze_technical_image(cap)

gr.Interface(
    fn=app,
    inputs=gr.Image(type="pil", label="Upload a technical image (Circuit Diagram, Simulation, etc.)"),
    outputs=[gr.Textbox(label="Visual Description (BLIP)"), gr.Textbox(label="Engineering Analysis")],
    title=APP_TITLE,
    description=f"Upload a photo — it gets captioned, then analyzed as a {ANALYSIS_STYLE}.",
).launch()
