#!/usr/bin/env python3
"""Simple Gradio chat UI for vLLM server."""

import argparse
import gradio as gr
from openai import OpenAI


def create_chat_interface(api_base: str, model_name: str):
    client = OpenAI(base_url=f"{api_base}/v1", api_key="dummy")
    
    def chat(message: str, history: list):
        messages = []
        # History format in newer Gradio: list of {"role": ..., "content": ...}
        for msg in history:
            if isinstance(msg, dict):
                messages.append({"role": msg["role"], "content": msg["content"]})
            else:
                # Old tuple format fallback: (user, assistant)
                user_msg, assistant_msg = msg
                messages.append({"role": "user", "content": user_msg})
                if assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        messages.append({"role": "user", "content": message})
        
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        
        partial_message = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                partial_message += chunk.choices[0].delta.content
                yield partial_message
    
    demo = gr.ChatInterface(
        chat,
        title=f"🦙 {model_name}",
        description=f"Chat with your trained model via vLLM",
        examples=[
            "What is the capital of France?",
            "Write a short poem about coding.",
            "Explain machine learning in simple terms.",
        ],
    )
    return demo


def main():
    parser = argparse.ArgumentParser(description="Chat UI for vLLM")
    parser.add_argument("--api-base", default="http://localhost:8000", help="vLLM API base URL")
    parser.add_argument("--model", default="dolci-instruct-sft-hf", help="Model name")
    parser.add_argument("--port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    args = parser.parse_args()
    
    demo = create_chat_interface(args.api_base, args.model)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()

