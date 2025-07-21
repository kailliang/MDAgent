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

## Detailed Analysis: process_advanced_query Function

The `process_advanced_query` function (`utils.py:761-852`) implements the most complex medical decision-making system using Multidisciplinary Teams (MDTs). This function represents the pinnacle of collaborative medical expertise for the most challenging cases.

### Function Signature and Purpose
```python
def process_advanced_query(question, model_to_use):
    # Returns: ({0.0: final_response_str}, sample_input_tokens, sample_output_tokens)
```

### Three-Step MDT Architecture

#### Step 1: Multidisciplinary Team Recruitment (`utils.py:766-791`)
**Purpose**: Organize specialized medical teams with distinct roles and expertise areas

**Key Components**:

1. **MDT Recruiter Creation** (`utils.py:771-772`):
   ```python
   recruiter_agent_mdt = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
   recruiter_agent_mdt.chat(recruit_prompt)
   ```
   - Uses user-specified model for recruitment decisions
   - Specialized in multidisciplinary team formation and medical workflow design

2. **Team Structure Requirements** (`utils.py:774-775`):
   ```python
   num_teams_to_form = 3        # Fixed number of teams
   num_agents_per_team = 3      # Fixed team size
   ```
   - **Team 1**: Initial Assessment Team (IAT) - Primary evaluation and triage
   - **Team 2**: Specialist Team - Domain-specific expertise  
   - **Team 3**: Final Review and Decision Team (FRDT) - Integration and final judgment

3. **Structured Team Recruitment** (`utils.py:777`):
   - Provides detailed example format with specific roles and hierarchies
   - Requires each team to have a designated Lead member
   - Mandates inclusion of IAT and FRDT teams
   - Uses structured prompt engineering for consistent team formation

4. **Team Parsing and Instantiation** (`utils.py:779-791`):
   ```python
   groups_text_list = [group_text.strip() for group_text in recruited_mdt_text.split("Group") if group_text.strip()]
   group_strings_list = ["Group " + group_text for group_text in groups_text_list]
   
   for i1, group_str_item in enumerate(group_strings_list):
       parsed_group_struct = parse_group_info(group_str_item)
       group_instance_obj = Group(parsed_group_struct['group_goal'], 
                                 parsed_group_struct['members'], 
                                 question, examplers=None, model_info=model_to_use)
   ```
   - Uses `parse_group_info()` to extract team goals and member specifications
   - Creates Group instances for each MDT with specialized objectives
   - No few-shot learning used (direct reasoning approach)

#### Step 2: MDT Internal Interactions and Assessments (`utils.py:793-823`)
**Purpose**: Execute parallel team assessments with role-specific processing

**Team Classification Logic** (`utils.py:798-811`):
```python
for group_obj in group_instances_list:
    group_goal_lower = group_obj.goal.lower()
    if 'initial' in group_goal_lower or 'iat' in group_goal_lower:
        # Process Initial Assessment Team
        init_assessment_text = group_obj.interact(comm_type='internal')
        initial_assessments_list.append([group_obj.goal, init_assessment_text])
    elif 'review' in group_goal_lower or 'decision' in group_goal_lower or 'frdt' in group_goal_lower:
        # Process Final Review/Decision Team  
        decision_text = group_obj.interact(comm_type='internal')
        final_review_decisions_list.append([group_obj.goal, decision_text])
    else:
        # Process Specialist Team
        assessment_text = group_obj.interact(comm_type='internal')
        other_mdt_assessments_list.append([group_obj.goal, assessment_text])
```

**Internal Group Interaction Mechanism** (Group.interact(), `utils.py:269-326`):

1. **Leadership Identification** (`utils.py:271-285`):
   ```python
   lead_member = None
   assist_members = []
   for member in self.members:
       if 'lead' in member_role.lower():
           lead_member = member
       else:
           assist_members.append(member)
   ```
   - Automatically identifies Lead member by role name
   - Falls back to first member if no explicit lead found

2. **Task Delegation** (`utils.py:287-307`):
   ```python
   delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. 
   You have the following assistant clinicians who work for you:'''
   delivery = lead_member.chat(delivery_prompt)
   ```
   - Lead member formulates investigation strategy
   - Delegates specific tasks to assistant clinicians
   - Error handling with fallback to assistant members

3. **Investigation Execution** (`utils.py:309-320`):
   ```python
   investigations = []
   if assist_members:
       for a_mem in assist_members:
           investigation = a_mem.chat("You are in a medical group where the goal is to {}. 
           Your group lead is asking for the following investigations:\n{}\n\n
           Please remind your expertise and return your investigation summary...".format(self.goal, delivery))
           investigations.append([a_mem.role, investigation])
   ```
   - Each assistant member performs specialized investigations
   - Results collected and structured by role

4. **Synthesis and Decision** (`utils.py:322-326`):
   ```python
   investigation_prompt = f"""The gathered investigation from your assistant clinicians 
   (or your own initial assessment if working alone) is as follows:\n{gathered_investigation}.
   \n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""
   response = lead_member.chat(investigation_prompt)
   ```
   - Lead member synthesizes all investigation results
   - Makes team-level medical decision based on comprehensive assessment

**Report Compilation** (`utils.py:813-823`):
```python
compiled_report_str = "[Initial Assessments]\n"
# ... compile initial assessments ...
compiled_report_str += "[Specialist Team Assessments]\n" 
# ... compile specialist assessments ...
compiled_report_str += "[Final Review Team Decisions (if any before overall final decision)]\n"
# ... compile final review decisions ...
```
- Structures all team outputs into comprehensive medical report
- Maintains clear separation between assessment types
- Preserves team attribution for transparency

#### Step 3: Final Decision from Overall Coordinator (`utils.py:825-834`)
**Purpose**: Integrate all MDT assessments into unified medical decision

**Overall Coordinator Creation** (`utils.py:828-829`):
```python
final_decision_agent = Agent(instruction=final_decision_prompt, role='Overall Coordinator', model_info=model_to_use)
final_decision_agent.chat(final_decision_prompt)
```
- Creates dedicated coordinator agent with overarching medical authority
- Uses same model as processing teams for consistency

**Final Decision Process** (`utils.py:831-834`):
```python
final_decision_dict_adv = final_decision_agent.temp_responses(f"""Combined MDT Investigations and Conclusions:
\n{compiled_report_str}\n\nBased on all the above, what is the final answer to the original medical query?
\nQuestion: {question}\nYour answer should be in the format: Answer: A) Example Answer""", img_path=None)

final_response_str = final_decision_dict_adv.get(0.0, "Error: Final coordinator failed to provide a decision.")
```
- Temperature=0.0 for deterministic final decision
- Synthesizes comprehensive MDT report into structured answer
- Enforces specific answer format for consistency

### Token Usage Tracking (`utils.py:836-851`)

Comprehensive token calculation across all teams and agents:

```python
# MDT Recruiter agent
recruiter_usage = recruiter_agent_mdt.get_token_usage()
sample_input_tokens += recruiter_usage['input_tokens']
sample_output_tokens += recruiter_usage['output_tokens']

# All team members across all groups
for group in group_instances_list:
    for member in group.members:
        member_usage = member.get_token_usage()
        sample_input_tokens += member_usage['input_tokens']
        sample_output_tokens += member_usage['output_tokens']

# Final coordinator agent
final_agent_usage = final_decision_agent.get_token_usage()
sample_input_tokens += final_agent_usage['input_tokens']
sample_output_tokens += final_agent_usage['output_tokens']
```

### Key Design Features

1. **Hierarchical Team Structure**: Clear lead-assistant relationships within each team
2. **Specialized Team Roles**: Distinct purposes for assessment, specialization, and review
3. **Parallel Processing**: Teams operate independently before integration
4. **Comprehensive Documentation**: Full audit trail from team formation to final decision
5. **Error Resilience**: Fallback mechanisms for team leadership and communication failures
6. **Structured Integration**: Systematic compilation and synthesis of team outputs

### Advanced Processing Characteristics

- **Scalability**: Supports variable team compositions and specializations
- **Modularity**: Teams can be added/removed without affecting core workflow
- **Quality Assurance**: Multiple layers of review and validation
- **Transparency**: Complete traceability from individual expert opinions to final decision
- **Flexibility**: Adapts team formation to specific medical query requirements
- **Resource Efficiency**: Parallel team processing minimizes sequential dependencies

### Team Interaction Patterns

1. **Intra-team Communication**: Lead delegates → Assistants investigate → Lead synthesizes
2. **Inter-team Isolation**: Teams work independently to prevent bias propagation
3. **Central Coordination**: Overall coordinator integrates all team perspectives
4. **Hierarchical Decision Flow**: Individual → Team → Inter-team → Final coordinator

This advanced processing mode represents the most sophisticated medical decision-making approach in the system, mimicking real-world multidisciplinary medical team consultations with systematic knowledge integration and expert consensus building.