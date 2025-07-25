# MDAgents: Adaptive Collaboration of LLMs for Medical Decision-Making

This is a research implementation of "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making" from NeurIPS 2024. The system uses multiple Large Language Models (LLMs) to collaboratively solve medical questions through three adaptive difficulty levels.

## 🏆 Performance Achievements

### Latest Results (Ultimate Branch)
- **Overall System Accuracy**: 84.04% across all difficulty levels
- **Basic Mode**: 87.05% accuracy (195/224 samples) - exceptional performance
- **Intermediate Mode**: 78.67% accuracy (59/75 samples) with multi-agent collaboration  
- **Advanced Mode**: 75.76% accuracy (25/33 samples) with multi-disciplinary teams
- **Token Efficiency**: 8.67M total tokens (5.75M input + 2.93M output)

## 🚀 Major Improvements & Achievements

### Enhanced Architecture
- ✅ **Basic Mode Optimization**: Upgraded from single agent to 3-expert + arbitrator system
- ✅ **JSON-based Communication**: All processing modes use structured JSON responses
- ✅ **Enhanced Evaluation System**: Multi-pattern answer extraction for diverse LLM response formats
- ✅ **Robust Parsing**: "Answer: X" pattern detection and parse error tracking
- ✅ **Word Limits**: Enforced response limits (50-300 words) for efficiency

### System Reliability
- ✅ **Multi-layer Error Handling**: JSON parsing → regex extraction → text fallback → default responses
- ✅ **Production-Ready Output**: Debug controls for clean production vs verbose development output
- ✅ **Comprehensive Token Tracking**: Monitored usage for all agents, recruiters, and coordinators

## 📋 Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Processing Modes](#processing-modes)
- [Evaluation](#evaluation)
- [Development](#development)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- API keys for OpenAI and/or Google Gemini

### Setup
1. Clone the repository:
```bash
git clone <repository-url>
cd MDAgents
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create environment file:
```bash
# Create .env file with your API keys
echo "openai_api_key=your_openai_api_key_here" > .env
echo "genai_api_key=your_gemini_api_key_here" >> .env
```

4. Activate virtual environment (if using):
```bash
source venv/bin/activate
```

## 🚀 Quick Start

### Basic Usage
```bash
python main.py --dataset medqa --model gemini-2.5-flash-lite-preview-06-17 --difficulty adaptive --num_samples 1
```

### Available Models
- **Gemini**: `gemini-2.0-flash`, `gemini-2.5-flash`, `gemini-2.5-flash-lite-preview-06-17`
- **OpenAI**: `gpt-4o-mini`, `gpt-4.1-mini`

### Difficulty Modes
- `adaptive`: System determines complexity automatically
- `basic`: 3-expert recruitment + arbitrator processing
- `intermediate`: Expert collaboration with multi-round debate
- `advanced`: Multi-disciplinary team approach

## 🏗️ Architecture

### Core Components

1. **Agent Class** (`utils.py:53+`): LLM interaction wrapper supporting OpenAI and Gemini
2. **Group Class** (`utils.py:190+`): Manages collaborative medical expert teams
3. **Processing Pipeline** (`main.py`): Main execution with difficulty assessment and routing

### Data Flow
```
Input JSONL → Difficulty Assessment → Processing Mode → Expert Collaboration → Final Answer → JSON Output
```

## 🧠 Processing Modes

### Basic Processing (87.05% accuracy)
- **Expert Recruitment**: 3 independent medical specialists
- **Independent Analysis**: Structured JSON responses with reasoning
- **Arbitrator Decision**: Synthesizes expert opinions into final answer
- **Efficiency**: Highly token-efficient with exceptional performance

### Intermediate Processing (78.67% accuracy)
- **Expert Recruitment**: 3 experts with hierarchical relationships
- **Multi-round Debate**: Collaborative discussion with adaptive participation
- **JSON Communication**: Structured participation decisions
- **Moderated Decision**: Team consensus synthesis

### Advanced Processing (75.76% accuracy)
- **MDT Formation**: Multidisciplinary teams with specialized roles
- **Parallel Assessment**: Independent team work then coordination
- **Overall Coordinator**: Final decision synthesis
- **Comprehensive Approach**: Full team-based medical decision making

## 📊 Evaluation

### Run Evaluation
```bash
# Evaluate system output and generate CSV reports
python evaluate_text_output.py

# Split test data by difficulty for analysis
python split_test_data.py

# Extract specific questions by ID for debugging
python extract_by_question_id.py
```

### Key Features
- **Multi-pattern Answer Extraction**: Handles diverse LLM response formats
- **Parse Error Tracking**: Identifies and tracks parsing failures
- **CSV Export**: Detailed evaluation reports with per-difficulty metrics
- **Answer Validation**: Robust extraction from malformed responses

### File Structure
```
output/           # JSON output files from system runs
evaluation/       # CSV evaluation reports with accuracy metrics
data/medqa/       # Test datasets split by difficulty level
```

## 🔧 Development

### System Controls (main.py)
```python
# Processing skip switches
SKIP_BASIC = False          # Skip basic difficulty questions
SKIP_INTERMEDIATE = False   # Skip intermediate difficulty questions  
SKIP_ADVANCED = False       # Skip advanced difficulty questions

# Debug controls (utils.py)
SHOW_INTERACTION_TABLE = False  # Display agent interaction tables
```

### Key Configuration
Update evaluation script filenames as needed:
```python
# In evaluate_text_output.py
input_filename = 'output/inter_json_adaptive_332samples.json'
output_filename = 'evaluation/inter_json_adaptive_332samples.csv'
```

### Common Patterns
- **JSON-First Communication**: Structured agent interactions
- **Multi-layer Error Handling**: Comprehensive fallback mechanisms
- **Temperature Control**: 0.0 for decisions, 0.7 for collaboration
- **Token Efficiency**: Word limits across all processing modes

## 📈 Performance Tracking

### Recent Achievements
- **Outstanding Basic Mode**: 87.05% accuracy through enhanced expert recruitment
- **Robust Evaluation**: Fixed parsing for complex responses and diverse answer formats
- **System Reliability**: Comprehensive error handling for malformed LLM responses
- **Performance Tracking**: Detailed CSV reporting with per-difficulty accuracy metrics

### Output Format
```json
{
  "question": "Medical question text",
  "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
  "ground_truth": "A",
  "model_response": "Generated response",
  "determined_difficulty": "basic|intermediate|advanced",
  "token_usage": {"input": 1500, "output": 300}
}
```

## 🤝 Contributing

This is a research implementation. For questions or contributions, please refer to the original NeurIPS 2024 paper: "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making".

## 📄 License

Research implementation - please refer to original paper licensing terms.

---

**Current Status**: Ultimate branch with 84.04% overall accuracy and robust multi-agent medical decision-making capabilities.
