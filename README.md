# MDAgents: Adaptive Collaboration of LLMs for Medical Decision-Making

A multi-agent system developed and improved over the original NeurIPS 2024 paper, using collaborative Large Language Models to solve medical questions through adaptive difficulty assessment. The system automatically routes questions through three processing modes: basic (expert arbitration), intermediate (expert collaboration), and advanced (multi-disciplinary teams).

## 🏆 Performance Results

### Exceptional Results with Gemini-2.5-Flash
- **Overall Accuracy**: 93.67% - breakthrough performance
- **Basic Mode**: 93.29% accuracy (292/313 samples)
- **Intermediate Mode**: 100.00% accuracy (17/17 samples) - perfect performance
- **Advanced Mode**: 100.00% accuracy (2/2 samples) - perfect performance
- **Token Efficiency**: 7.23M total tokens

### Cost-Effective Results with Gemini-2.5-Flash-Lite
- **Overall Accuracy**: 84.04% across all difficulty levels  
- **Basic Mode**: 87.05% accuracy with 3-expert + arbitrator system
- **Intermediate Mode**: 78.67% accuracy with multi-agent collaboration
- **Advanced Mode**: 75.76% accuracy with multi-disciplinary teams
- **Token Efficiency**: 8.67M total tokens

## ⭐ Ultimate Branch Achievements

### Performance vs Original Paper (MedQA, 332 samples)
- **This Implementation (Gemini-2.5-Flash)**: 93.67% accuracy, 7.23M tokens - **breakthrough results**
- **This Implementation (Gemini-2.5-Flash-Lite)**: 84.04% accuracy, 8.67M tokens - **cost-effective excellence**
- **Original Paper Method**: 79.45% accuracy, 67.44M tokens
- **Best Improvement**: +14.22% accuracy with **89.3% fewer tokens** using Gemini-2.5-Flash
- **Cost-Effective Improvement**: +4.59% accuracy with **87.1% fewer tokens** using Gemini-2.5-Flash-Lite

### Key Innovations
- **Cost-Effective Excellence**: Achieved superior performance using the affordable **Gemini-2.5-Flash-Lite** model
- **Enhanced Architecture**: Optimized agent architecture and communication mechanisms for improved collaboration
- **JSON-Based Communication**: Implemented structured agent interactions with robust error handling
- **Production-Ready**: Multi-pattern answer extraction with comprehensive fallback mechanisms
- **Token Optimization**: Efficient processing with enforced word limits (50-300 words per response)

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
echo "openai_api_key=your_key" > .env
echo "genai_api_key=your_key" >> .env
```

### Usage
```bash
python main.py --dataset medqa --model gemini-2.5-flash-lite-preview-06-17 --difficulty adaptive --num_samples 1
```

### Models & Modes
- **Models**: Gemini (2.0/2.5-flash), OpenAI (gpt-4o-mini, gpt-4.1-mini)
- **Modes**: `adaptive`, `basic`, `intermediate`, `advanced`

## 🧠 Processing Modes

**Basic (87.05%)**: 3 independent medical experts + arbitrator synthesis
**Intermediate (78.67%)**: Expert collaboration with multi-round debate
**Advanced (75.76%)**: Multi-disciplinary teams with coordinator

## 📊 Evaluation
```bash
python evaluate_text_output.py  # Generate CSV reports
python split_test_data.py       # Split by difficulty
```

## 🏗️ Architecture
- **Agent Class** (`utils.py:53+`): LLM wrapper for OpenAI/Gemini
- **Group Class** (`utils.py:190+`): Collaborative expert teams
- **Pipeline** (`main.py`): Difficulty assessment and routing

## 🔧 Configuration
```python
# main.py - Processing controls
SKIP_BASIC = False
SKIP_INTERMEDIATE = False  
SKIP_ADVANCED = False

# utils.py - Debug output
SHOW_INTERACTION_TABLE = False
```
