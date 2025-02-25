import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
import json
import argparse

class PixelGenie:
    def __init__(self, model_name="mistralai/Mistral-7B-Instruct"):
        """Initialize the PixelGenie AI Model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

    def generate_response(self, prompt, max_length=500):
        """Generate AI-powered responses based on input prompt."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)
    
    def execute_code(self, code, language="python"):
        """Execute given code in the specified programming language."""
        if language == "python":
            try:
                exec_globals = {}
                exec(code, exec_globals)
                return exec_globals
            except Exception as e:
                return str(e)
        elif language == "bash":
            try:
                result = subprocess.run(code, shell=True, capture_output=True, text=True)
                return result.stdout if result.returncode == 0 else result.stderr
            except Exception as e:
                return str(e)
        else:
            return "Unsupported language."
    
    def optimize_code(self, code):
        """Provide suggestions for optimizing the given code."""
        prompt = f"Optimize the following code for better performance:\n{code}"
        return self.generate_response(prompt)
    
    def explain_code(self, code):
        """Explain the given code in simple terms."""
        prompt = f"Explain the following code in simple terms:\n{code}"
        return self.generate_response(prompt)
    
    def suggest_best_practices(self, topic):
        """Suggest best practices for a given topic."""
        prompt = f"What are the best practices for {topic}?"
        return self.generate_response(prompt)
    
    def interactive_cli(self):
        """Launch a command-line interface for user interaction."""
        print("Welcome to PixelGenie CLI! Type 'exit' to quit.")
        while True:
            user_input = input("\nAsk PixelGenie: ")
            if user_input.lower() == "exit":
                print("Goodbye!")
                break
            response = self.generate_response(user_input)
            print("\nResponse:")
            print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PixelGenie - AI-powered coding assistant")
    parser.add_argument("--mode", type=str, choices=["cli", "explain", "optimize", "best-practices"], default="cli", help="Mode of operation")
    parser.add_argument("--code", type=str, help="Code snippet to process")
    parser.add_argument("--topic", type=str, help="Topic for best practices suggestion")
    
    args = parser.parse_args()
    assistant = PixelGenie()
    
    if args.mode == "cli":
        assistant.interactive_cli()
    elif args.mode == "explain" and args.code:
        print(assistant.explain_code(args.code))
    elif args.mode == "optimize" and args.code:
        print(assistant.optimize_code(args.code))
    elif args.mode == "best-practices" and args.topic:
        print(assistant.suggest_best_practices(args.topic))
    else:
        print("Invalid arguments. Run with --help for usage information.")
