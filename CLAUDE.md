# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making" from NeurIPS 2024. The system uses multiple Large Language Models (LLMs) to collaboratively solve medical questions through three adaptive difficulty levels: basic (single agent), intermediate (expert collaboration), and advanced (multi-disciplinary teams).

## Development Setup

### Environment Setup
1. Install Python dependencies: `pip install -r requirements.txt`
2. Create a `.env` file with API keys:
   - `openai_api_key=your_openai_api_key_here`
   - `genai_api_key=your_gemini_api_key_here`
3. Activate virtual environment if using one: `source venv/bin/activate`

### Running the System
Basic usage:
```bash
python main.py --dataset medqa --model gemini-2.0-flash --difficulty adaptive --num_samples 1
```

Available models:
- `gemini-2.0-flash`, `gemini-2.5-flash` (requires `genai_api_key`)
- `gpt-4o-mini`, `gpt-4.1-mini` (requires `openai_api_key`)

Difficulty modes:
- `adaptive`: System determines complexity automatically
- `basic`: Single agent processing
- `intermediate`: Expert collaboration with debate
- `advanced`: Multi-disciplinary team approach

## Architecture

### Core Components

1. **Agent Class** (`utils.py:20-187`): Wrapper for LLM interactions supporting both OpenAI and Gemini models
2. **Group Class** (`utils.py:190-268`): Manages collaborative medical expert teams
3. **Processing Pipeline** (`main.py:58-121`): Main execution loop with difficulty assessment and routing

### Processing Modes

- **Basic Processing** (`utils.py:415-446`): Single medical agent with few-shot examples
- **Intermediate Processing** (`utils.py:448-665`): 
  - Expert recruitment and hierarchy establishment
  - Multi-round collaborative debate between agents
  - Moderated final decision making
- **Advanced Processing** (`utils.py:667-737`): 
  - Multi-disciplinary team (MDT) formation
  - Internal team assessments
  - Cross-team coordination and final decision

### Data Structure

- Input: JSONL files in `data/{dataset}/` (test.jsonl, train.jsonl)
- Output: JSON files in `output/` with format: `{model}_{dataset}_{difficulty}_{samples}samples.json`
- Each result includes: question, options, ground truth, model response, and determined difficulty

### Key Utility Functions

- `setup_model()`: Configures API clients based on model type
- `determine_difficulty()`: Uses LLM to assess question complexity for adaptive mode
- `load_data()`: Loads test questions and exemplars from dataset files
- `create_question()`: Formats questions with randomized multiple choice options

## Common Patterns

- All LLM interactions go through the Agent class for consistent error handling
- Unicode cleaning is performed on API responses to handle encoding issues
- Temperature settings: 0.0 for deterministic responses, 0.7 for creative collaboration
- Retry logic implemented for API failures with exponential backoff
- Results are saved incrementally to prevent data loss on interruption