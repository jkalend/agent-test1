from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain.tools import Tool
from transformers import pipeline
from langgraph.prebuilt import create_react_agent
import re

# funnily enough, the tools are too simple for modern LLMs
def calculator_tool(expression: str) -> str:
    """Simple calculator tool that can evaluate basic math expressions"""
    try:
        # Eval is not the best way to do this, but it's a simple way to do it
        if re.match(r'^[0-9+\-*/().\s]+$', expression):
            result = eval(expression)
            return f"The result is: {result}"
        else:
            return "Invalid expression. Please use only numbers and basic operators (+, -, *, /, (), .)"
    except Exception as e:
        return f"Error calculating: {str(e)}"

def text_length_tool(text: str) -> str:
    """Tool to count characters and words in text"""
    char_count = len(text)
    word_count = len(text.split())
    return f"Character count: {char_count}, Word count: {word_count}"

def upper_case_tool(text: str) -> str:
    """Tool to convert text to uppercase"""
    return text.upper()

# Define tools
tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Useful for doing math calculations. Input should be a mathematical expression like '2+2' or '10*5'"
    ),
    Tool(
        name="TextAnalyzer", 
        func=text_length_tool,
        description="Useful for analyzing text length. Input should be the text you want to analyze."
    ),
    Tool(
        name="UpperCase",
        func=upper_case_tool,
        description="Useful for converting text to uppercase. Input should be the text you want to convert."
    )
]

# Create the Hugging Face pipeline
print("Loading Hugging Face model...")

try:
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    pipeline_kwargs = {
        "model": "Qwen/Qwen3-4B",
        "torch_dtype": torch.float16 if torch.cuda.is_available() else "auto",
        "device_map": "auto",
        "max_new_tokens": 4096,
        "temperature": 0.1,
        "do_sample": True,
        "top_p": 0.9,
        "repetition_penalty": 1.1,
        "return_full_text": False,
    }
    
    hf_pipeline = pipeline("text-generation", **pipeline_kwargs)
    print("✅ Qwen model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Create LangChain LLM from the pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Wrap in ChatHuggingFace for tool binding capability
chat_model = ChatHuggingFace(llm=llm)

# For create_react_agent, we need a simpler system prompt
system_prompt = """You are a helpful assistant that can use tools to answer questions.

You have access to the following tools:
- Calculator: For mathematical calculations
- TextAnalyzer: For analyzing text length (character and word count)
- UpperCase: For converting text to uppercase

When you need to use a tool, think about which tool would be most appropriate and use it.
When you invoke a tool, give the user the result of the tool call, without modifying it.
Always provide a clear and helpful final answer to the user's question."""

# Create the agent using the new langgraph approach
agent_executor = create_react_agent(chat_model, tools, prompt=system_prompt)

def run_agent():
    """Main function to run the agent interactively"""
    print("\n🤖 Simple LangChain + HuggingFace Agent")
    print("Available tools: Calculator, TextAnalyzer, UpperCase")
    print("Type 'quit' to exit")
    print("\nTry these examples:")
    print("- 'What is 15 * 7?'")
    print("- 'How many characters are in this sentence?'")
    print("- 'Convert hello world to uppercase'")
    print("- 'What is 100 / 4?'")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
                
            if not user_input:
                continue
                
            print("\nAgent is thinking...")
            
            try:
                response = agent_executor.invoke({"messages": [("user", user_input)]})
                
                if 'messages' in response:
                    output = response['messages'][-1].content.strip()
                    if output:
                        print(f"\nAgent: {output}\n")
                    else:
                        print("\nAgent: [No response generated]\n")
                else:
                    print(f"\nAgent: {str(response)}\n")
                    
            except Exception as agent_error:
                print(f"\nAgent error: {agent_error}")
                print("The model might be having trouble with tool usage.")
                print("Try rephrasing your question or use one of the example prompts above.\n")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    run_agent() 
