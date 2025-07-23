# Simple LangChain Agent Exploration

A basic exploration into LangChain fundamentals using LangGraph's ReAct agent framework with HuggingFace models.

## Overview

This project explores the basics of:
- LangChain agent construction
- Tool binding and execution
- HuggingFace model integration with LangGraph

## Model Testing Notes

Uses Qwen/Qwen3-4B which provides reliable tool calling capabilities. Smaller models, like DialoGPT, often have issues with tool calling reliability and instruction following, based on my experience.

## Tools Included

- Calculator for math operations
- Text analyzer for counting characters and words  
- Text case converter to uppercase 
