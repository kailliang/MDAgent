# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of "MDAgents: An Adaptive Collaboration of LLMs for Medical Decision-Making" from NeurIPS 2024. The system uses multiple Large Language Models (LLMs) to collaboratively solve medical questions through three adaptive difficulty levels: basic (expert arbitration), intermediate (expert collaboration), and advanced (multi-disciplinary teams).

## Current Status (Ultimate Branch)

### Performance Results (Latest - Updated)
- **Basic Mode**: 83.18% accuracy (178/214 correct) - significantly improved from 71.60%
- **Intermediate Mode**: 78.87% accuracy (56/71 correct) with multi-agent collaboration
- **Advanced Mode**: 78.72% accuracy (37/47 correct) with MDT approach
- **Overall Adaptive**: 81.63% accuracy (271/332 correct) across all difficulty levels
- **Branch Status**: `ultimate` branch contains the most optimized and stable version

### Latest Improvements
- ✅ **JSON-based Communication**: All processing modes use structured JSON responses
- ✅ **Enhanced Basic Mode**: 3-expert recruitment + arbitrator system (was single agent)
- ✅ **Improved Evaluation System**: Enhanced parsing for diverse response formats
- ✅ **Debug Controls**: `SHOW_INTERACTION_TABLE` for production vs development output
- ✅ **Processing Controls**: Skip switches (`SKIP_BASIC`, `SKIP_INTERMEDIATE`, `SKIP_ADVANCED`)
- ✅ **Word Limits**: Enforced response limits (50-300 words) for efficiency
- ✅ **Robust Error Handling**: Multiple JSON parsing strategies with fallbacks

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
- `basic`: 3-expert recruitment + arbitrator processing
- `intermediate`: Expert collaboration with multi-round debate
- `advanced`: Multi-disciplinary team approach

### Evaluation and Testing
```bash
# Evaluate system output and generate CSV reports
python evaluate_text_output.py

# Split test data by difficulty for analysis
python split_test_data.py

# Extract specific questions by ID for debugging
python extract_by_question_id.py
```

Key evaluation files:
- `output/`: JSON output files from system runs (e.g., `medqa_adaptive_332samples.json`)
- `evaluation/`: CSV evaluation reports with accuracy metrics (e.g., `medqa_adaptive_332samples.csv`)
- `data/medqa/`: Test datasets split by difficulty level

**Important**: The `evaluate_text_output.py` script contains configurable input/output filenames at the top:
```python
input_filename = 'output/inter_json_adaptive_332samples.json'  # Update as needed
output_filename = 'evaluation/inter_json_adaptive_332samples.csv'  # Update as needed
```

## Architecture

### Core Components

1. **Agent Class** (`utils.py:20-187`): Wrapper for LLM interactions supporting both OpenAI and Gemini models
2. **Group Class** (`utils.py:190-268`): Manages collaborative medical expert teams
3. **Processing Pipeline** (`main.py:58-121`): Main execution loop with difficulty assessment and routing

### Processing Modes (Updated Architecture)

- **Basic Processing** (`utils.py:540-752`): 
  - **Expert Recruitment**: 3 independent medical specialists with equal authority
  - **Independent Analysis**: Each expert provides structured JSON response with reasoning
  - **Arbitrator Decision**: Medical arbitrator synthesizes expert opinions into final answer
  - **Performance**: 83.18% accuracy, highly token-efficient
  
- **Intermediate Processing** (`utils.py:755-945`): 
  - **Expert Recruitment**: 3 experts with hierarchical relationships
  - **Multi-round Debate**: 3 rounds × 3 turns collaborative discussion
  - **JSON Communication**: Structured participation decisions and expert selection
  - **Moderated Decision**: Final moderator synthesizes team consensus
  - **Word Limits**: 50-200 word responses for efficiency
  
- **Advanced Processing** (`utils.py:947-1031`): 
  - **MDT Formation**: 3 multidisciplinary teams (IAT, Specialist, FRDT)
  - **JSON Team Structure**: Structured team and member definitions
  - **Parallel Assessment**: Teams work independently then coordinate
  - **Overall Coordinator**: Final decision synthesis with JSON analysis format
  - Internal team assessments
  - Cross-team coordination and final decision

### Data Structure

- Input: JSONL files in `data/{dataset}/` (test.jsonl, train.jsonl)
- Output: JSON files in `output/` with format: `{model}_{dataset}_{difficulty}_{samples}samples.json`
- Each result includes: question, options, ground truth, model response, determined difficulty, and token usage

### System Controls (main.py)

```python
# Processing skip switches - control which difficulty levels to process
SKIP_BASIC = False          # Skip basic difficulty questions
SKIP_INTERMEDIATE = False   # Skip intermediate difficulty questions  
SKIP_ADVANCED = False       # Skip advanced difficulty questions

# Debug controls (utils.py)
SHOW_INTERACTION_TABLE = False  # Display agent interaction tables in intermediate mode
```

### Key Utility Functions

- `setup_model()`: Configures API clients based on model type
- `determine_difficulty()`: Uses LLM with JSON format to assess question complexity for adaptive mode
- `load_data()`: Loads test questions and exemplars from dataset files
- `create_question()`: Formats questions with randomized multiple choice options

### Evaluation System (`evaluate_text_output.py`)

Enhanced parsing system with multi-pattern answer extraction:

```python
def extract_final_answer_or_answer(text):
    # Handles multiple response formats:
    # 1. "Answer: C" patterns (common in majority_vote responses)
    # 2. "B) Normal hemoglobin..." patterns  
    # 3. "(A) Some answer" parentheses formats
    # 4. Parse error cases marked as 'X'
    # 5. Various structured answer formats
```

**Key Features**:
- **Multi-pattern Recognition**: Handles diverse LLM response formats
- **Parse Error Tracking**: Identifies and tracks parsing failures
- **CSV Export**: Generates detailed evaluation reports with per-difficulty metrics
- **Answer Validation**: Robust extraction from complex majority_vote responses

**Critical Parsing Improvements**:
- Early "Answer: X" detection for long majority_vote responses
- Enhanced word boundary handling for answer extraction
- Fallback patterns for malformed or incomplete responses
- Support for various parentheses and formatting styles

## Common Patterns

- **JSON-First Communication**: All agent interactions use structured JSON formats with regex parsing and fallbacks
- **Multi-layer Error Handling**: JSON parsing → regex extraction → text fallback → default responses
- **Temperature Control**: 0.0 for deterministic final decisions, 0.7 for creative expert collaboration
- **Token Efficiency**: Word limits enforced across all processing modes (50-300 words)
- **Comprehensive Tracking**: Token usage monitored for all agents, recruiters, and coordinators
- **Production-Ready Output**: Debug controls provide clean output for production vs verbose for development

## Development Priorities

### Completed ✅
- **Basic Mode Optimization**: Enhanced from single agent to 3-expert + arbitrator system
- **JSON Communication**: Structured responses across all processing modes
- **Error Resilience**: Multi-layer parsing with comprehensive fallbacks
- **Performance Measurement**: Accurate evaluation scripts for new JSON formats
- **Debug Controls**: Clean production output with optional verbose debugging

### Current Focus
- **Performance Optimization**: Recent improvements achieved 81.63% overall accuracy
- **Evaluation System Refinement**: Enhanced parsing handles diverse response formats
- **Response Format Standardization**: Improved JSON parsing and fallback mechanisms
- **Token Cost Analysis**: Efficiency optimization while maintaining accuracy

### Recent Achievements (Latest Updates)
- **Enhanced Basic Mode**: Improved from 71.60% to 83.18% accuracy through better expert recruitment
- **Robust Evaluation**: Fixed parsing for complex majority_vote responses and diverse answer formats
- **System Reliability**: Comprehensive error handling for malformed LLM responses
- **Performance Tracking**: Detailed CSV reporting with per-difficulty accuracy metrics

## Detailed Analysis: process_intermediate_query Function

The `process_intermediate_query` function (`utils.py:514-743`) implements a sophisticated multi-agent collaborative medical decision-making system. This function represents the core of the intermediate difficulty processing mode.

### Function Signature and Purpose
```python
def process_intermediate_query(question, examplers_data, model_to_use, args):
    # Returns: (final_decision_dict, sample_input_tokens, sample_output_tokens)
```

### Three-Phase Processing Architecture

#### Phase 1: Expert Recruitment (`utils.py:519-578`)
**Purpose**: Dynamically assemble a team of medical experts based on question content

**Key Components**:
1. **Recruiter Agent Creation** (`utils.py:522-523`):
   ```python
   recruiter_agent = Agent(instruction=recruit_prompt, role='recruiter', 
                          model_info='gemini-2.5-flash-lite-preview-06-17')
   ```
   - Uses dedicated Gemini model for recruitment decisions
   - Specialized in medical expert selection and team composition

2. **Expert Recruitment Process** (`utils.py:525-526`):
    - 这一步是关键的招募，很重要。下一步改。
    - Recruits exactly 5 medical experts with diverse specializations
    - Requests hierarchy specification between experts (e.g., "Pediatrician > Cardiologist")
    - Uses structured prompt to ensure consistent expert information format

3. **Data Parsing and Agent Instantiation** (`utils.py:528-566`):
   ```python
   agents_data_parsed = [(info[0], info[1]) if len(info) > 1 else (info[0], None) 
                         for info in agents_info_raw]
   ```
   - Parses expert descriptions and hierarchy relationships
   - Creates individual Agent instances for each recruited expert
   - Maintains `agent_dict` for role-based access and `medical_agents_list` for iteration

**Key Variables**:
- `num_experts_to_recruit = 5`: Fixed team size for consistency
- `agents_data_parsed`: List of (expert_description, hierarchy) tuples
- `medical_agents_list`: All expert agent instances
- `agent_dict`: Role-name to agent mapping

#### Phase 2: Collaborative Decision Making (`utils.py:580-705`)
**Purpose**: Multi-round collaborative debate with adaptive participation

**Debate Structure**:
```python
num_rounds = 5      # Maximum debate rounds
num_turns = 5       # Maximum turns per round
```

**Sub-phases**:

1. **Initial Opinion Collection** (`utils.py:605-609`):
   ```python
   for agent_role_key, agent_instance in agent_dict.items():
       opinion = agent_instance.chat(f'''Please return your answer to the medical query...''')
       initial_report_str += f"({agent_role_key.lower()}): {opinion}\n"
       round_opinions_log[1][agent_role_key.lower()] = opinion
   ```
   - Each expert provides independent initial assessment
   - No few-shot examples used (direct reasoning approach)
   - Opinions stored in `round_opinions_log[1]`

2. **Multi-Round Debate Loop** (`utils.py:612-675`):
   
   **Round Structure**:
   - Each round has up to 5 turns for agent interactions
   - Agents decide autonomously whether to participate each turn
   - Participation based on reviewing current collective opinions
   
   **Turn Mechanics** (`utils.py:627-653`):
   ```python
   participate_decision = agent_instance_loop.chat(
       f"Given the opinions from other medical experts... please indicate whether you want to talk to any expert (yes/no).")
   ```
   - **Participation Decision**: Each agent evaluates if they need to communicate
   - **Target Selection**: Participating agents choose specific experts to address
   - **Message Exchange**: Source agent sends opinion/question to target agent
   - **Interaction Logging**: All communications recorded in `interaction_log`

   **Interaction Tracking**:
   ```python
   interaction_log = {f'Round {r}': {f'Turn {t}': 
       {f'Agent {s}': {f'Agent {trg}': None for trg in range(1, num_active_agents + 1)} 
        for s in range(1, num_active_agents + 1)} 
       for t in range(1, num_turns + 1)} 
      for r in range(1, num_rounds + 1)}
   ```

3. **Opinion Evolution** (`utils.py:664-669`):
   ```python
   if r_idx < num_rounds:
       next_round_opinions = {}
       for agent_idx_collect, agent_instance_collect in enumerate(medical_agents_list):
           opinion_prompt = f"Reflecting on the discussions in Round {r_idx}, what is your current answer/opinion..."
           response = agent_instance_collect.chat(opinion_prompt)
           next_round_opinions[agent_instance_collect.role.lower()] = response
       round_opinions_log[r_idx+1] = next_round_opinions
   ```
   - Opinions updated after each round based on discussions
   - Tracks opinion evolution throughout debate process

4. **Termination Conditions**:
   - **No Participation**: If no agents participate in a turn/round, advance or terminate
   - **Natural Conclusion**: Agents organically reach consensus or stable disagreement
   - **Round Limit**: Maximum 5 rounds to prevent infinite loops

5. **Interaction Visualization** (`utils.py:677-704`):
   ```python
   myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i%len(agent_emoji)]})" 
                                for i in range(num_active_agents)])
   ```
   - Generates interaction matrix showing communication patterns
   - Uses emojis for agent identification
   - Displays bidirectional communication indicators

#### Phase 3: Final Decision (`utils.py:706-718`)
**Purpose**: Synthesize all expert opinions into final medical decision

**Moderator System**:
```python
moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", 
                  "Moderator", model_info=model_to_use)
```

**Decision Process**:
```python
moderator_decision_dict = moderator.temp_responses(
    f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote or synthesizing the best response...Agent Opinions:\n{final_answer_map}\n\nQuestion: {question}")
```

- **Opinion Synthesis**: Moderator reviews all expert final opinions
- **Decision Methods**: Majority voting or expert opinion synthesis
- **Structured Output**: Ensures consistent answer format
- **Temperature=0.0**: Uses deterministic response for final decision

### Token Usage Tracking (`utils.py:723-742`)

Comprehensive token usage calculation across all components:

```python
# Recruiter agent tokens
recruiter_usage = recruiter_agent.get_token_usage()
sample_input_tokens += recruiter_usage['input_tokens']
sample_output_tokens += recruiter_usage['output_tokens']

# All medical expert agents
for agent in medical_agents_list:
    agent_usage = agent.get_token_usage()
    sample_input_tokens += agent_usage['input_tokens']
    sample_output_tokens += agent_usage['output_tokens']

# Moderator agent
moderator_usage = moderator.get_token_usage()
sample_input_tokens += moderator_usage['input_tokens']
sample_output_tokens += moderator_usage['output_tokens']

# Summarizer agents (created during rounds)
for summarizer_agent in summarizer_agents_list:
    summarizer_usage = summarizer_agent.get_token_usage()
    sample_input_tokens += summarizer_usage['input_tokens']
    sample_output_tokens += summarizer_usage['output_tokens']
```

### Key Design Features

1. **Adaptive Participation**: Agents autonomously decide when to contribute based on evolving discussion
2. **Hierarchy Support**: Accommodates expert hierarchies and independent specialists
3. **Complete Audit Trail**: All interactions logged for analysis and debugging
4. **Scalable Architecture**: Supports variable team sizes and round limits
5. **Error Resilience**: Graceful handling of agent failures and communication errors
6. **Visualization Ready**: Provides structured data for interaction analysis

### Performance Characteristics

- **Token Efficiency**: Minimizes redundant communications through adaptive participation
- **Quality Assurance**: Multi-round validation improves decision accuracy
- **Transparency**: Complete interaction logging for result interpretation
- **Flexibility**: Adapts team composition to question complexity

This implementation represents a sophisticated approach to multi-agent medical decision-making, balancing collaborative discussion with computational efficiency.



## Comprehensive Analysis: process_advanced_query Execution Flow

Based on detailed code analysis, here is the complete execution sequence and LLM call pattern for the Advanced processing mode:

### Current Implementation Status
**Configuration**: 2 teams, 2 members each (reduced from original 3x3 design)
**Performance**: 64.52% accuracy (40/62 samples) - lowest among the three modes
**Key Issue**: Simplified team structure may not capture sufficient expertise diversity

### Detailed LLM Call Sequence

#### STEP 1: MDT Recruitment (`utils.py:964-1038`)

**Call 1: Recruiter Initialization**
```python
recruiter_agent_mdt = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
recruiter_agent_mdt.chat(recruit_prompt)
```
- **Prompt**: `"You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."`
- **Purpose**: Initialize recruiter agent identity

**Call 2: Team Recruitment with JSON Structure**
```python
recruited_mdt_response = recruiter_agent_mdt.chat(recruitment_prompt)
```
- **Full Prompt**:
```
Question: {question}

You should organize 2 MDTs with different specialties or purposes and each MDT should have 2 clinicians. Return your recruitment plan in JSON format with the following structure:

{
  "teams": [
    {
      "team_id": 1,
      "team_name": "Initial Assessment Team (IAT)",
      "members": [
        {
          "member_id": 1,
          "role": "Otolaryngologist (ENT Surgeon) (Lead)",
          "expertise_description": "Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage."
        },
        {
          "member_id": 2,
          "role": "General Surgeon",
          "expertise_description": "Provides additional surgical expertise and supports in the overall management of thyroid surgery complications."
        }
      ]
    }
  ]
}

You must include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. Each team should have exactly 2 members with one designated as Lead. Return only valid JSON without markdown code blocks or explanations.
```
- **Expected Output**: JSON with 2 teams (IAT + FRDT), 2 members each
- **Critical Constraint**: Must include both IAT and FRDT teams

#### STEP 2: Team Member Initialization (`utils.py:1007-1020`)

**For Each Team Member (4 total agents created)**:
```python
_agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), 
               role=member_info['role'], model_info=model_to_use)
_agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
```
- **Prompt Pattern**: `"You are a {role} who {expertise_description}."`
- **Example**: `"You are a Otolaryngologist (ENT Surgeon) (Lead) who specializes in ear, nose, and throat surgery, including thyroidectomy."`

#### STEP 3: MDT Internal Interactions (`utils.py:1045-1058`)

**For Each Team (2 teams), Group.interact('internal') is called:**

**Sub-call 3.1: Team Lead Task Delegation**
```python
delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:
{assistant_list}

Now, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians (if any), or outline your approach. Strictly limit your response with no more than 30 words. 
Question: {self.question}'''

delivery = lead_member.chat(delivery_prompt)
```
- **Purpose**: Lead member assigns investigation tasks
- **Word Limit**: 30 words maximum

**Sub-call 3.2: Assistant Member Investigations (1 per team)**
```python
investigation_prompt = "You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information. Strictly limit your response with no more than 50 words.".format(self.goal, delivery)

investigation = assistant_member.chat(investigation_prompt)
```
- **Purpose**: Assistant performs assigned investigation
- **Word Limit**: 50 words maximum

**Sub-call 3.3: Lead Member Synthesis and Team Decision**
```python
investigation_prompt = f"""The gathered investigation from your assistant clinicians (or your own initial assessment if working alone) is as follows:
{gathered_investigation}.

Now, return your answer to the medical query among the option provided. Limit your response with no more than 100 words.

Question: {self.question}"""

response = lead_member.chat(investigation_prompt)
```
- **Purpose**: Lead synthesizes investigations into team decision
- **Word Limit**: 100 words maximum

#### STEP 4: Overall Coordination (`utils.py:1072-1078`)

**Call 4.1: Coordinator Initialization**
```python
final_decision_agent = Agent(instruction=final_decision_prompt, role='Overall Coordinator', model_info=model_to_use)
final_decision_agent.chat(final_decision_prompt)
```
- **Prompt**: `"You are an experienced medical coordinator. Given the investigations and conclusions from multiple multidisciplinary teams (MDTs), please review them very carefully and return your final, consolidated answer to the medical query."`

**Call 4.2: Final Decision Synthesis**
```python
final_prompt = f"""Combined MDT Investigations and Conclusions:
{compiled_report_str}

Based on all the above, what is the final answer to the original medical query?
Question: {question}
Your answer should strictly be in the following format: 
Answer: A) Example Answer 
DO NOT include any explanation."""

final_decision_dict_adv = final_decision_agent.temp_responses(final_prompt, img_path=None)
```
- **Temperature**: 0.0 (deterministic)
- **Input**: Complete compiled report from all teams
- **Output Format**: Strictly "Answer: X) ..." format

### Total LLM Calls Per Query
1. **Recruiter**: 2 calls (initialization + recruitment)
2. **Team Members**: 4 agents × 2 calls = 8 calls (initialization + identity reinforcement)
3. **Team Internal Process**: 
   - 2 teams × 1 lead delegation call = 2 calls
   - 2 teams × 1 assistant investigation call = 2 calls 
   - 2 teams × 1 lead synthesis call = 2 calls
4. **Final Coordinator**: 2 calls (initialization + final decision)

**Total: ~18 LLM calls per advanced query**

### Data Flow and Team Classification

**Team Classification Logic**:
- **Initial Assessment Team (IAT)**: Keywords "initial" or "iat" in team goal
- **Final Review and Decision Team (FRDT)**: Keywords "review", "decision", or "frdt" in team goal  
- **Specialist Team**: Any other team not matching IAT/FRDT patterns

**Report Structure**:
```
[Initial Assessments]
Team 1 - {team_name}: {team_decision}

[Specialist Team Assessments]  
Team N - {team_name}: {team_decision}

[Final Review Team Decisions (if any before overall final decision)]
Team 2 - {team_name}: {team_decision}
```

### Performance Analysis Issues

**Identified Bottlenecks**:
1. **Limited Team Size**: Only 2 teams × 2 members = 4 total experts (vs 5 in intermediate mode)
2. **Rigid Structure**: Fixed IAT/FRDT requirement may not suit all medical questions
3. **Word Limits**: Strict 30/50/100 word limits may truncate important medical reasoning
4. **No Cross-Team Communication**: Teams work in isolation without collaboration
5. **Sequential Processing**: No parallel team discussions or consensus building

**Recommendations for Improvement**:
1. Increase team size back to 3×3 structure for better coverage
2. Allow dynamic team composition based on medical question complexity
3. Implement cross-team communication and consensus mechanisms
4. Remove or relax strict word limits for complex medical reasoning
5. Add specialized teams for specific medical domains (e.g., diagnostics, treatment, ethics)