

import gradio as gr
from transformers import pipeline
import torch

# Check if GPU is available
device = 0 if torch.cuda.is_available() else -1

# Load Mistral 7B Instruct model
try:
    model = pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.1",
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True,
        device=device
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you have enough GPU memory or use a smaller model")

SYSTEM_PROMPT = (
    "You are GeoGuide AI, a helpful and responsible geology assistant. "
    "Explain geology concepts clearly and simply for students and beginners. "
    "Use beginner-friendly language and avoid professional jargon. "
    "Focus on educational value only. Never provide advice for dangerous geological activities."
)

def chat(user_input, history):
    """Process user input and return updated chat history"""
    if not user_input.strip():
        return history, ""
    
    try:
        # Format prompt for Mistral Instruct format
        prompt = f"""[INST] {SYSTEM_PROMPT}
User: {user_input} [/INST]"""
        
        # Generate response
        response = model(prompt, max_new_tokens=300)
        
        # Extract the generated text after [/INST]
        full_text = response[0]["generated_text"]
        bot_reply = full_text.split("[/INST]")[-1].strip()
        
        # Clean up the response (remove extra prompts if any)
        if bot_reply.startswith("User:"):
            bot_reply = bot_reply.split("User:")[0].strip()
        
        history.append((user_input, bot_reply))
        return history, ""
    
    except Exception as e:
        error_msg = f"Error generating response: {str(e)}"
        history.append((user_input, error_msg))
        return history, ""

def preset_question(question, history):
    """Handle preset question button clicks"""
    return chat(question, history)

def clear_chat():
    """Clear chat history"""
    return []

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    
    # Header
    gr.Markdown("# 🌍 GeoGuide AI - Geology Learning Chatbot")
    gr.Markdown(
        "An interactive geology learning chatbot powered by Mistral 7B. "
        "Perfect for students and beginners!\n\n"
        "⚠️ **Disclaimer:** Educational use only. Not a substitute for professional geological advice."
    )
    
    # Chat display
    chatbot = gr.Chatbot(
        label="Chat History",
        height=400,
        show_label=True
    )
    
    # State to track conversation history
    state = gr.State([])
    
    # Input section
    with gr.Row():
        user_input = gr.Textbox(
            placeholder="Ask a geology question... (e.g., 'What is basalt?')",
            show_label=False,
            lines=2
        )
    
    # Button row
    with gr.Row():
        send_btn = gr.Button("Send 📤", variant="primary")
        clear_btn = gr.Button("Clear Chat 🗑️")
    
    # Topic-wise buttons
    gr.Markdown("## 🪨 Quick Topic Buttons")
    with gr.Row():
        rocks_btn = gr.Button("Rocks & Types")
        minerals_btn = gr.Button("Minerals")
        earth_btn = gr.Button("Earth Processes")
    
    # Pre-written questions
    gr.Markdown("## 📌 Quick Questions")
    with gr.Row():
        q1 = gr.Button("What are the three types of rocks?")
        q2 = gr.Button("Explain the rock cycle")
    
    with gr.Row():
        q3 = gr.Button("Difference between igneous and sedimentary rocks")
        q4 = gr.Button("What is plate tectonics?")
    
    # Button click handlers - Send message
    send_btn.click(
        fn=chat,
        inputs=[user_input, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    # Enter key to send
    user_input.submit(
        fn=chat,
        inputs=[user_input, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    # Clear chat button
    clear_btn.click(
        fn=clear_chat,
        outputs=[state]
    ).then(
        fn=lambda: [],
        outputs=[chatbot]
    )
    
    # Topic button handlers
    rocks_btn.click(
        fn=preset_question,
        inputs=[gr.State("Explain the main types of rocks (igneous, sedimentary, metamorphic) with examples"), state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    minerals_btn.click(
        fn=preset_question,
        inputs=[gr.State("What are minerals? How are they classified and identified?"), state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    earth_btn.click(
        fn=preset_question,
        inputs=[gr.State("Explain major earth processes like volcanism, erosion, and plate tectonics"), state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    # Quick question handlers
    q1.click(
        fn=preset_question,
        inputs=[q1, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    q2.click(
        fn=preset_question,
        inputs=[q2, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    q3.click(
        fn=preset_question,
        inputs=[q3, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )
    
    q4.click(
        fn=preset_question,
        inputs=[q4, state],
        outputs=[state, user_input]
    ).then(
        fn=lambda h: h,
        inputs=[state],
        outputs=[chatbot]
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(share=True)
