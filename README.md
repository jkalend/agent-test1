# Simple LangChain Agent Exploration

A basic exploration into LangChain fundamentals using LangGraph's ReAct agent framework with HuggingFace models.

## Overview

This project explores the basics of:
- LangChain agent construction
- Tool binding and execution
- HuggingFace model integration with LangGraph

## Model Testing Notes

Uses Qwen/Qwen3-4B which provides reliable tool calling capabilities. Smaller models, like DialoGPT, often have issues with tool calling reliability and instruction following, based on my experience.

Curiously enough the qwen model is not able to output the JSON tool call result exactly as it is returned by the tool. It seems to add some extra formatting, even when it is prompted to NOT do so.

## Tools Included

- Calculator for math operations
- Text analyzer for counting characters and words  
- Text case converter to uppercase 
- JSON tool for creating or formatting JSON
