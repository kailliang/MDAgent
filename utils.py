import os
from dotenv import load_dotenv
import json
import random
from tqdm import tqdm
from prettytable import PrettyTable
from termcolor import cprint
from pptree import Node
import google.generativeai as genai
from openai import OpenAI
import requests
import time
import sys
import unicodedata # For normalization
import traceback   # For full traceback if other errors occur

# Load environment variables from .env file
load_dotenv()

# Global token usage tracking
GLOBAL_TOKEN_USAGE = {
    'total_input_tokens': 0,
    'total_output_tokens': 0,
    'sample_usage': []
}

def add_to_global_usage(input_tokens, output_tokens, sample_id=None):
    """Add token usage to global tracking"""
    GLOBAL_TOKEN_USAGE['total_input_tokens'] += input_tokens
    GLOBAL_TOKEN_USAGE['total_output_tokens'] += output_tokens
    if sample_id is not None:
        GLOBAL_TOKEN_USAGE['sample_usage'].append({
            'sample_id': sample_id,
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        })

def get_global_token_usage():
    """Get global token usage statistics"""
    return GLOBAL_TOKEN_USAGE.copy()

def reset_global_token_usage():
    """Reset global token usage counters"""
    GLOBAL_TOKEN_USAGE['total_input_tokens'] = 0
    GLOBAL_TOKEN_USAGE['total_output_tokens'] = 0
    GLOBAL_TOKEN_USAGE['sample_usage'] = []

class Agent:

    def __init__(self, instruction, role, examplers=None, model_info='gemini-2.5-flash-lite-preview-06-17', img_path=None):
        self.instruction = instruction
        self.role = role
        self.model_info = model_info
        self.img_path = img_path
        
        # Initialize token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        if self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            if 'genai_api_key' in os.environ:
                genai.configure(api_key=os.environ['genai_api_key'])
            else:
                raise ValueError("Gemini API key not configured. Set 'genai_api_key' in .env file or environment variables.")
            self.model = genai.GenerativeModel(self.model_info)
            self._chat = self.model.start_chat(history=[])
        elif self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:

            api_key = os.environ.get('openai_api_key')
            if not api_key:
                raise ValueError("OpenAI API key not found. Set 'openai_api_key' environment variable.")
            
            self.client = OpenAI(
                api_key=api_key, # API key is now clean
            )
            
            current_instruction_content = str(self.instruction)

            self.messages = [
                {"role": "system", "content": current_instruction_content},
            ]

            if examplers is not None:
                for exampler in examplers:
                    question = str(exampler.get('question', ''))
                    answer = str(exampler.get('answer', ''))
                    reason = str(exampler.get('reason', ''))

                    self.messages.append({"role": "user", "content": question})
                    self.messages.append({"role": "assistant", "content": answer + "\n\n" + reason})
        else:
            raise ValueError(f"Unsupported model_info: {self.model_info}")

    def _clean_problematic_unicode(self, text_content):
        # This function is still useful for cleaning message content before sending to LLMs,
        # especially if they are sensitive to certain Unicode characters or if you want to
        # normalize input. However, it wasn't the cause of the API key header issue.
        if not isinstance(text_content, str):
            if text_content is None:
                return ""
            try:
                text_content = str(text_content)
            except Exception:
                return ""

        try:
            normalized_text = unicodedata.normalize('NFKC', text_content)
        except TypeError:
            normalized_text = text_content
        
        normalized_text = normalized_text.replace('\u201c', '"').replace('\u201d', '"')
        normalized_text = normalized_text.replace('\u2018', "'").replace('\u2019', "'")
        normalized_text = normalized_text.replace('\u2013', '-').replace('\u2014', '--')

        ascii_bytes = normalized_text.encode('ascii', errors='replace')
        cleaned_string = ascii_bytes.decode('ascii')
        
        return cleaned_string
    
    def chat(self, message, img_path=None, chat_mode=True):
        if self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            for _ in range(10):
                try:
                    # Gemini expects UTF-8 strings for messages.
                    response = self._chat.send_message(str(message))
                    
                    # Track token usage for Gemini
                    if hasattr(response, 'usage_metadata'):
                        if hasattr(response.usage_metadata, 'prompt_token_count'):
                            self.total_input_tokens += response.usage_metadata.prompt_token_count
                        if hasattr(response.usage_metadata, 'candidates_token_count'):
                            self.total_output_tokens += response.usage_metadata.candidates_token_count
                    
                    return response.text
                except Exception as e:
                    # Safe cprint for error messages
                    error_str = f"Error communicating with Gemini: {e}"
                    safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                    cprint(safe_error_str, "red")
                    time.sleep(1) 
            return "Error: Failed to get response from Gemini after multiple retries."

        elif self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:
            # OpenAI also expects UTF-8 strings for message content.
            current_user_message_content = str(message)
            # cleaned_user_message_content = self._clean_problematic_unicode(current_user_message_content) # Potentially remove if not needed

            # self.messages contains original (or lightly cleaned) history
            api_call_messages = [msg.copy() for msg in self.messages]
            # Use original user message content if aggressive cleaning isn't needed
            api_call_messages.append({"role": "user", "content": current_user_message_content})
            
            # model_name = "gpt-4o-mini"

            try:
                response = self.client.chat.completions.create(
                    model=self.model_info,
                    messages=api_call_messages,
                    temperature=0.7
                )
                
                # Track token usage for OpenAI
                if hasattr(response, 'usage'):
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                raw_response_content = response.choices[0].message.content
                # cleaned_response_content = self._clean_problematic_unicode(raw_response_content) # Potentially remove

                # Add original user message and original assistant response to history
                self.messages.append({"role": "user", "content": current_user_message_content})
                self.messages.append({"role": "assistant", "content": raw_response_content})
                return raw_response_content

            except Exception as e: # Catch general exceptions; specific UnicodeEncodeError on headers is now fixed.
                error_str = f"Error communicating with OpenAI: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                cprint("FULL TRACEBACK for OpenAI Exception:", "red", force_color=True)
                traceback.print_exc()
                return f"Error: Failed to get response from OpenAI: {str(e)}"
        else:
            raise ValueError(f"Unsupported model_info in chat: {self.model_info}")

    def temp_responses(self, message, img_path=None):
        if self.model_info in ['gpt-4o-mini', 'gpt-4.1-mini']:
            current_user_message_content = str(message)
            # cleaned_user_message_content = self._clean_problematic_unicode(current_user_message_content) # Potentially remove

            api_call_messages = [msg.copy() for msg in self.messages]
            api_call_messages.append({"role": "user", "content": current_user_message_content})
            
            responses = {}
            # model_name = "gpt-4o-mini"

            try:
                response = self.client.chat.completions.create(
                    model=self.model_info,
                    messages=api_call_messages,
                    temperature=0.0
                )
                
                # Track token usage for OpenAI
                if hasattr(response, 'usage'):
                    self.total_input_tokens += response.usage.prompt_tokens
                    self.total_output_tokens += response.usage.completion_tokens
                
                raw_response_content = response.choices[0].message.content
                responses[0.0] = raw_response_content
                return responses

            except Exception as e:
                error_str = f"Error communicating with OpenAI in temp_responses: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                cprint("FULL TRACEBACK for OpenAI temp_responses Exception:", "red", force_color=True)
                traceback.print_exc()
                return {0.0: f"Error: Failed to get response from OpenAI: {str(e)}"}
        
        elif self.model_info in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
            try:
                response = self._chat.send_message(str(message))
                
                # Track token usage for Gemini
                if hasattr(response, 'usage_metadata'):
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        self.total_input_tokens += response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        self.total_output_tokens += response.usage_metadata.candidates_token_count
                
                return {0.0: response.text}
            except Exception as e:
                error_str = f"Error communicating with Gemini for temp_responses: {e}"
                safe_error_str = error_str.encode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace').decode(getattr(sys.stderr, 'encoding', 'utf-8') or 'utf-8', 'replace')
                cprint(safe_error_str, "red")
                return {0.0: "Error: Failed to get response from Gemini."}
        else:
            raise ValueError(f"Unsupported model_info in temp_responses: {self.model_info}")
    
    def get_token_usage(self):
        """Return current token usage for this agent"""
        return {
            'input_tokens': self.total_input_tokens,
            'output_tokens': self.total_output_tokens,
            'total_tokens': self.total_input_tokens + self.total_output_tokens
        }
    
    def reset_token_usage(self):
        """Reset token usage counters"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0


class Group:
    def __init__(self, goal, members, question, examplers=None, model_info='gemini-2.5-flash-lite-preview-06-17'):
        self.goal = goal
        self.members = []
        for member_info in members:
            # Group members use gpt-4o-mini
            _agent = Agent('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()), 
                           role=member_info['role'], 
                           model_info=model_info)
            _agent.chat('You are a {} who {}.'.format(member_info['role'], member_info['expertise_description'].lower()))
            self.members.append(_agent)
        self.question = question
        self.examplers = examplers

    def interact(self, comm_type, message=None, img_path=None):
        if comm_type == 'internal':
            lead_member = None
            assist_members = []
            for member in self.members:
                member_role = member.role

                if 'lead' in member_role.lower():
                    lead_member = member
                else:
                    assist_members.append(member)

            if lead_member is None:
                if not assist_members:
                    cprint("Warning: Group has no members or no identifiable lead/assistant.", "red")
                    return "Error: Group configuration issue."
                lead_member = assist_members[0]
            
            delivery_prompt = f'''You are the lead of the medical group which aims to {self.goal}. You have the following assistant clinicians who work for you:'''
            if assist_members:
                for a_mem in assist_members:
                    delivery_prompt += "\n{}".format(a_mem.role)
            else:
                delivery_prompt += "\nYou are working independently or with a predefined protocol to address the goal."

            delivery_prompt += "\n\nNow, given the medical query, provide a short answer to what kind investigations are needed from each assistant clinicians (if any), or outline your approach.\nQuestion: {}".format(self.question)
            
            try:
                delivery = lead_member.chat(delivery_prompt)
            except Exception as e:
                cprint(f"Error during lead_member chat: {e}", "red")
                if assist_members and lead_member != assist_members[0]:
                    try:
                        delivery = assist_members[0].chat(delivery_prompt)
                    except Exception as e2:
                        cprint(f"Error during fallback assistant chat: {e2}", "red")
                        return "Error: Could not get delivery from group lead or assistants."
                else:
                    return "Error: Could not get delivery from group lead."

            investigations = []
            if assist_members:
                for a_mem in assist_members:
                    investigation = a_mem.chat("You are in a medical group where the goal is to {}. Your group lead is asking for the following investigations:\n{}\n\nPlease remind your expertise and return your investigation summary that contains the core information.".format(self.goal, delivery))
                    investigations.append([a_mem.role, investigation])
            
            gathered_investigation = ""
            if investigations:
                for investigation_item in investigations:
                    gathered_investigation += "[{}]\n{}\n".format(investigation_item[0], investigation_item[1])
            else:
                gathered_investigation = delivery

            # Direct reasoning without few-shot examples
            investigation_prompt = f"""The gathered investigation from your assistant clinicians (or your own initial assessment if working alone) is as follows:\n{gathered_investigation}.\n\nNow, return your answer to the medical query among the option provided.\n\nQuestion: {self.question}"""

            response = lead_member.chat(investigation_prompt)
            return response

        elif comm_type == 'external':
            return "External communication not implemented."
        else:
            return "Unknown communication type."

def parse_hierarchy(info, emojis):
    moderator = Node('moderator (\U0001F468\u200D\u2696\uFE0F)')
    agents = [moderator]
    count = 0
    
    def get_emoji(index):
        return emojis[index % len(emojis)]
    
    for expert, hierarchy_str in info:
        try:
            expert_name = expert.split('-')[0].split('.')[1].strip()
        except:
            expert_name = expert.split('-')[0].strip()
        
        if hierarchy_str is None:
            hierarchy_str = 'Independent'
        
        if 'independent' not in hierarchy_str.lower():
            parent_name = hierarchy_str.split(">")[0].strip()
            child_name = hierarchy_str.split(">")[1].strip()

            parent_node_found = False
            for agent_node in agents:
                if agent_node.name.split("(")[0].strip().lower() == parent_name.strip().lower():
                    child_agent_node = Node("{} ({})".format(child_name, get_emoji(count)), agent_node)
                    agents.append(child_agent_node)
                    parent_node_found = True
                    break
            if not parent_node_found:
                cprint(f"Warning: Parent '{parent_name}' for child '{child_name}' not found. Adding child to moderator.", "yellow")
                agent_node = Node("{} ({})".format(expert_name, get_emoji(count)), moderator)
                agents.append(agent_node)
        else:
            agent_node = Node("{} ({})".format(expert_name, get_emoji(count)), moderator)
            agents.append(agent_node)
        count += 1
    return agents

def parse_group_info(group_info):
    lines = group_info.split('\n')
    parsed_info = {
        'group_goal': '',
        'members': []
    }
    if not lines or not lines[0]:
        return parsed_info

    goal_parts = lines[0].split('-', 1)
    if len(goal_parts) > 1:
        parsed_info['group_goal'] = goal_parts[1].strip()
    else:
        parsed_info['group_goal'] = goal_parts[0].strip()
    
    for line in lines[1:]:
        if line.startswith('Member'):
            member_parts = line.split(':', 1)
            if len(member_parts) < 2: continue

            member_role_description_str = member_parts[1].split('-', 1)
            
            member_role = member_role_description_str[0].strip()
            member_expertise = member_role_description_str[1].strip() if len(member_role_description_str) > 1 else 'General expertise'
            
            parsed_info['members'].append({
                'role': member_role,
                'expertise_description': member_expertise
            })
    return parsed_info

def setup_model(model_name):
    # Normalize model_name to avoid issues with whitespace/case
    original_model_name = model_name
    model_name = str(model_name).strip()
    print(f"[DEBUG] setup_model received model_name: '{original_model_name}' (normalized: '{model_name}')")
    if model_name in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
        if 'genai_api_key' in os.environ:
            genai.configure(api_key=os.environ['genai_api_key'])
            return True
        else:
            cprint("Error: 'genai_api_key' not found for Gemini setup.", "red")
            return False
    elif model_name in ['gpt-4o-mini', 'gpt-4.1-mini']:
        if 'openai_api_key' in os.environ:
            return True
        else:
            cprint("Error: 'openai_api_key' not found for OpenAI setup.", "red")
            return False
    else:
        supported = ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17', 'gpt-4o-mini', 'gpt-4.1-mini']
        raise ValueError(f"Unsupported model for setup: '{original_model_name}'. Supported models: {supported}")

def load_data(dataset):
    test_qa = []
    examplers = []
    base_data_path = os.path.join(os.path.dirname(__file__), 'data')

    test_path = os.path.join(base_data_path, dataset, 'test.jsonl')
    print(f"[DEBUG] Loading test data from: {test_path}")
    try:
        with open(test_path, 'r', encoding='utf-8') as file:
            for line in file:
                test_qa.append(json.loads(line))
    except FileNotFoundError:
        cprint(f"Error: Test data file not found at {test_path}", "red")

    train_path = os.path.join(base_data_path, dataset, 'train.jsonl')
    try:
        with open(train_path, 'r', encoding='utf-8') as file:
            for line in file:
                examplers.append(json.loads(line))
    except FileNotFoundError:
        cprint(f"Error: Train data file (exemplars) not found at {train_path}", "red")
    print(f"[DEBUG] test_qa loaded: {len(test_qa)}")
    return test_qa, examplers

def create_question(sample, dataset):
    if dataset == 'medqa':
        question = sample['question'] + " Options: "
        options = []
        for k, v in sample['options'].items():
            options.append("({}) {}".format(k, v))
        random.shuffle(options)
        question += " ".join(options)
        return question, None
    return sample.get('question', "No question provided in sample."), None

def determine_difficulty(question, difficulty):
    if difficulty != 'adaptive':
        return difficulty, 0, 0  # Return difficulty with zero token usage for non-adaptive
    
    difficulty_prompt = f"""Now, given the medical query as below, you need to decide the difficulty/complexity of it:
{question}

Please indicate the difficulty/complexity of the medical query among below options:
1) basic: a single medical agent can output an answer.
2) intermediate: number of medical experts with different expertise should dicuss and make final decision.
3) advanced: multiple teams of clinicians from different departments need to collaborate with each other to make final decision."""
    
    medical_agent = Agent(instruction='You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.', role='medical expert', model_info='gemini-2.5-flash-lite-preview-06-17')
    medical_agent.chat('You are a medical expert who conducts initial assessment and your job is to decide the difficulty/complexity of the medical query.')
    response = medical_agent.chat(difficulty_prompt)

    # Get token usage from the difficulty determination agent
    difficulty_agent_usage = medical_agent.get_token_usage()
    difficulty_input_tokens = difficulty_agent_usage['input_tokens']
    difficulty_output_tokens = difficulty_agent_usage['output_tokens']

    if 'basic' in response.lower() or '1)' in response.lower():
        return 'basic', difficulty_input_tokens, difficulty_output_tokens
    elif 'intermediate' in response.lower() or '2)' in response.lower():
        return 'intermediate', difficulty_input_tokens, difficulty_output_tokens
    elif 'advanced' in response.lower() or '3)' in response.lower():
        return 'advanced', difficulty_input_tokens, difficulty_output_tokens
    else:
        cprint(f"Warning: Could not parse difficulty from response: '{response}'. Defaulting to intermediate.", "yellow")
        return 'intermediate', difficulty_input_tokens, difficulty_output_tokens

def process_basic_query(question, examplers_data, model_to_use, args):
    import re
    import json
    
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    # Create single agent without few-shot examples - direct reasoning
    single_agent = Agent(
        instruction='You are a medical expert that answers multiple choice questions about medical knowledge.', 
        role='medical expert', 
        examplers=None,  # No few-shot examples
        model_info=model_to_use
    )
    single_agent.chat('You are a medical expert that answers multiple choice questions about medical knowledge.')
    
    # Enhanced prompt for JSON output with strict formatting
    prompt = """You are a medical expert. Analyze the following multiple choice question and provide your response in exactly this JSON format:

{{
  "reasoning": "Your step-by-step medical analysis in no more than 500 words",
  "answer": "X"
}}

**Requirements:**
- Answer must be a single letter (A, B, C, D, E, etc.) corresponding to one of the provided options
- Return ONLY the JSON, no other text

**Question:** {}
"""
    
    max_retries = 2
    temperatures = [0.0, 0.3, 0.7]  # Progressive temperature adjustment
    final_decision_dict = None
    
    for attempt in range(max_retries + 1):
        try:
            current_temp = temperatures[min(attempt, len(temperatures)-1)]
            cprint(f"Attempt {attempt + 1} with temperature {current_temp}", "cyan")
            
            # Create a new agent for each retry to reset temperature
            if attempt > 0:
                single_agent = Agent(
                    instruction='You are a medical expert that answers multiple choice questions about medical knowledge.', 
                    role='medical expert', 
                    examplers=None,
                    model_info=model_to_use
                )
                single_agent.chat('You are a medical expert that answers multiple choice questions about medical knowledge.')
            
            # Modify temp_responses to accept temperature parameter
            if model_to_use in ['gpt-4o-mini', 'gpt-4.1-mini']:
                # For OpenAI, we need to modify the API call with custom temperature
                current_user_message_content = str(prompt.format(question))
                api_call_messages = [msg.copy() for msg in single_agent.messages]
                api_call_messages.append({"role": "user", "content": current_user_message_content})
                
                response = single_agent.client.chat.completions.create(
                    model=single_agent.model_info,
                    messages=api_call_messages,
                    temperature=current_temp
                )
                
                # Track token usage
                if hasattr(response, 'usage'):
                    single_agent.total_input_tokens += response.usage.prompt_tokens
                    single_agent.total_output_tokens += response.usage.completion_tokens
                
                raw_response = response.choices[0].message.content
                response_dict = {0.0: raw_response}
                
            elif model_to_use in ['gemini-2.5-flash', 'gemini-2.5-flash-lite-preview-06-17']:
                # For Gemini, use the existing temp_responses method
                response_dict = single_agent.temp_responses(prompt.format(question), img_path=None)
                raw_response = response_dict.get(0.0, "")
            else:
                raise ValueError(f"Unsupported model: {model_to_use}")
            
            if not raw_response or "Error:" in raw_response:
                cprint(f"Attempt {attempt + 1}: API call failed - {raw_response}", "yellow")
                continue
            
            # Try to extract and validate JSON
            try:
                # Look for JSON pattern in the response
                json_match = re.search(r'\{[^{}]*"reasoning"\s*:[^{}]*"answer"\s*:\s*"[^"]*"[^{}]*\}', raw_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    parsed_json = json.loads(json_str)
                    
                    reasoning = parsed_json.get("reasoning", "").strip()
                    answer = parsed_json.get("answer", "").strip().upper()
                    
                    # Validate answer format (single letter)
                    if len(answer) == 1 and answer.isalpha():
                        final_decision_dict = {
                            "reasoning": reasoning,
                            "answer": answer
                        }
                        cprint(f"Attempt {attempt + 1}: Valid JSON response received - Answer: {answer}", "green")
                        break
                    else:
                        cprint(f"Attempt {attempt + 1}: Invalid answer format '{answer}'", "yellow")
                        continue
                        
                else:
                    cprint(f"Attempt {attempt + 1}: No valid JSON found in response", "yellow")
                    continue
                    
            except json.JSONDecodeError as e:
                cprint(f"Attempt {attempt + 1}: JSON parsing failed - {str(e)}", "yellow")
                continue
            
        except Exception as e:
            cprint(f"Attempt {attempt + 1}: Exception occurred - {str(e)}", "red")
            continue
    
    # If all attempts failed, create fallback response
    if final_decision_dict is None:
        cprint("All attempts failed, creating fallback response", "red")
        final_decision_dict = {
            "reasoning": "Unable to process question after multiple attempts",
            "answer": "A"
        }
    
    # Calculate token usage for this sample (only single agent)
    single_agent_usage = single_agent.get_token_usage()
    sample_input_tokens = single_agent_usage['input_tokens']
    sample_output_tokens = single_agent_usage['output_tokens']
    
    return final_decision_dict, sample_input_tokens, sample_output_tokens

def process_intermediate_query(question, examplers_data, model_to_use, args):
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    cprint("[INFO] Step 1. Expert Recruitment", 'yellow', attrs=['blink'])
    recruit_prompt = "You are an experienced medical expert who recruits a group of experts with diverse identity and ask them to discuss and solve the given medical query."
    
    recruiter_agent = Agent(instruction=recruit_prompt, role='recruiter', model_info='gemini-2.5-flash-lite-preview-06-17')
    recruiter_agent.chat(recruit_prompt)
    
    num_experts_to_recruit = 5
    recruited_text = recruiter_agent.chat(f"Question: {question}\nYou can recruit {num_experts_to_recruit} experts in different medical expertise. Considering the medical question and the options for the answer, what kind of experts will you recruit to better make an accurate answer?\nAlso, you need to specify the communication structure between experts (e.g., Pulmonologist == Neonatologist == Medical Geneticist == Pediatrician > Cardiologist), or indicate if they are independent.\n\nFor example, if you want to recruit five experts, you answer can be like:\n1. Pediatrician - Specializes in the medical care of infants, children, and adolescents. - Hierarchy: Independent\n2. Cardiologist - Focuses on the diagnosis and treatment of heart and blood vessel-related conditions. - Hierarchy: Pediatrician > Cardiologist\n3. Pulmonologist - Specializes in the diagnosis and treatment of respiratory system disorders. - Hierarchy: Independent\n4. Neonatologist - Focuses on the care of newborn infants, especially those who are born prematurely or have medical issues at birth. - Hierarchy: Independent\n5. Medical Geneticist - Specializes in the study of genes and heredity. - Hierarchy: Independent\n\nPlease answer in above format, and do not include your reason.")

    agents_info_raw = [agent_info.split(" - Hierarchy: ") for agent_info in recruited_text.split('\n') if agent_info.strip()]
    agents_data_parsed = [(info[0], info[1]) if len(info) > 1 else (info[0], None) for info in agents_info_raw]

    agent_emoji = ['\U0001F468\u200D\u2695\uFE0F', '\U0001F468\U0001F3FB\u200D\u2695\uFE0F', '\U0001F469\U0001F3FC\u200D\u2695\uFE0F', '\U0001F469\U0001F3FB\u200D\u2695\uFE0F', '\U0001f9d1\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3ff\u200D\u2695\uFE0F', '\U0001f468\U0001f3fd\u200D\u2695\uFE0F', '\U0001f9d1\U0001f3fd\u200D\u2695\uFE0F', '\U0001F468\U0001F3FD\u200D\u2695\uFE0F']
    random.shuffle(agent_emoji)

    if num_experts_to_recruit > len(agent_emoji):
        agent_emoji.extend(['\U0001F9D1'] * (num_experts_to_recruit - len(agent_emoji)))

    hierarchy_agents_nodes = parse_hierarchy(agents_data_parsed, agent_emoji)

    agent_list_str = ""
    for i, agent_tuple in enumerate(agents_data_parsed):
        try:
            agent_desc_parts = str(agent_tuple[0]).split('-', 1)
            role_part = agent_desc_parts[0].split('.', 1)[-1].strip().lower() if '.' in agent_desc_parts[0] else agent_desc_parts[0].strip().lower()
            description = agent_desc_parts[1].strip().lower() if len(agent_desc_parts) > 1 else "No description"
            agent_list_str += f"Agent {i+1}: {role_part} - {description}\n"
        except Exception as e:
            cprint(f"Error parsing agent data for list: {agent_tuple} - {e}", "red")
            agent_list_str += f"Agent {i+1}: Error parsing agent info\n"

    agent_dict = {}
    medical_agents_list = []
    for agent_tuple in agents_data_parsed:
        try:
            agent_desc_parts = str(agent_tuple[0]).split('-', 1)
            agent_role_parsed = agent_desc_parts[0].split('.', 1)[-1].strip().lower() if '.' in agent_desc_parts[0] else agent_desc_parts[0].strip().lower()
            description_parsed = agent_desc_parts[1].strip().lower() if len(agent_desc_parts) > 1 else "No description"
        except Exception as e:
            cprint(f"Error parsing agent data for instantiation: {agent_tuple} - {e}", "red")
            continue
        
        inst_prompt = f"""You are a {agent_role_parsed} who {description_parsed}. Your job is to collaborate with other medical experts in a team."""
        _agent_instance = Agent(instruction=inst_prompt, role=agent_role_parsed, model_info=model_to_use)
        
        _agent_instance.chat(inst_prompt)
        agent_dict[agent_role_parsed] = _agent_instance
        medical_agents_list.append(_agent_instance)

    for idx, agent_tuple in enumerate(agents_data_parsed):
        try:
            emoji = agent_emoji[idx % len(agent_emoji)]
            agent_name_part = str(agent_tuple[0]).split('-')[0].strip()
            agent_desc_part = str(agent_tuple[0]).split('-', 1)[1].strip() if '-' in str(agent_tuple[0]) else "N/A"
            print(f"Agent {idx+1} ({emoji} {agent_name_part}): {agent_desc_part}")
        except IndexError:
             print(f"Agent {idx+1} ({agent_emoji[idx % len(agent_emoji)]}): {agent_tuple[0]}")
        except Exception as e:
            cprint(f"Error printing agent info: {agent_tuple} - {e}", "red")

    print()
    cprint("[INFO] Step 2. Collaborative Decision Making", 'yellow', attrs=['blink'])
    cprint("[INFO] Step 2.1. Hierarchy Selection", 'yellow', attrs=['blink'])
    if hierarchy_agents_nodes:
        try:
            from pptree import print_tree
            print_tree(hierarchy_agents_nodes[0], horizontal=False)
        except ImportError:
            cprint("pptree not installed or print_tree not found. Skipping hierarchy print.", "yellow")
        except Exception as e:
            cprint(f"Error printing tree: {e}", "red")

    print()

    num_rounds = 5
    num_turns = 5
    num_active_agents = len(medical_agents_list)

    interaction_log = {f'Round {r}': {f'Turn {t}': {f'Agent {s}': {f'Agent {trg}': None for trg in range(1, num_active_agents + 1)} for s in range(1, num_active_agents + 1)} for t in range(1, num_turns + 1)} for r in range(1, num_rounds + 1)}

    cprint("[INFO] Step 2.2. Participatory Debate", 'yellow', attrs=['blink'])

    round_opinions_log = {r: {} for r in range(1, num_rounds+1)}
    initial_report_str = ""
    summarizer_agents_list = []  # Track all summarizer agents for token usage
    
    for agent_role_key, agent_instance in agent_dict.items():
        # Direct reasoning without few-shot examples
        opinion = agent_instance.chat(f'''Please return your answer to the medical query among the option provided.\n\nQuestion: {question}\n\nYour answer should be like below format.\n\nAnswer: ''', img_path=None)
        initial_report_str += f"({agent_role_key.lower()}): {opinion}\n"
        round_opinions_log[1][agent_role_key.lower()] = opinion

    final_answer_map = None
    for r_idx in range(1, num_rounds+1):
        print(f"== Round {r_idx} ==")
        round_name_str = f"Round {r_idx}"
        
        summarizer_agent = Agent(instruction="You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.", role="medical assistant", model_info=model_to_use)
        summarizer_agent.chat("You are a medical assistant who excels at summarizing and synthesizing based on multiple experts from various domain experts.")
        summarizer_agents_list.append(summarizer_agent)  # Track for token usage
        
        current_assessment_str = "".join(f"({k.lower()}): {v}\n" for k, v in round_opinions_log[r_idx].items())

        num_participated_this_round = 0
        for t_idx in range(num_turns):
            turn_name_str = f"Turn {t_idx + 1}"
            print(f"|_{turn_name_str}")

            num_participated_this_turn = 0
            for agent_idx, agent_instance_loop in enumerate(medical_agents_list):
                context_for_participation_decision = current_assessment_str

                participate_decision = agent_instance_loop.chat(f"Given the opinions from other medical experts in your team (see below), please indicate whether you want to talk to any expert (yes/no).\n\nOpinions:\n{context_for_participation_decision}")
                
                if 'yes' in participate_decision.lower().strip():                
                    chosen_expert_indices_str = agent_instance_loop.chat(f"Enter the number of the expert you want to talk to (1-{num_active_agents}):\n{agent_list_str}\nFor example, if you want to talk with Agent 1. Pediatrician, return just 1. If you want to talk with more than one expert, please return 1,2 (comma-separated) and don't return the reasons.")
                    
                    chosen_expert_indices = [int(ce.strip()) for ce in chosen_expert_indices_str.replace('.', ',').split(',') if ce.strip().isdigit()]

                    for target_expert_idx_chosen in chosen_expert_indices:
                        if 1 <= target_expert_idx_chosen <= num_active_agents:
                            target_agent_actual_idx = target_expert_idx_chosen - 1
                            specific_question_to_expert = agent_instance_loop.chat(f"Please remind your medical expertise and then leave your opinion/question for an expert you chose (Agent {target_expert_idx_chosen}. {medical_agents_list[target_agent_actual_idx].role}). You should deliver your opinion once you are confident enough and in a way to convince other expert, with a short reason.")
                            
                            source_emoji = agent_emoji[agent_idx % len(agent_emoji)]
                            target_emoji = agent_emoji[target_agent_actual_idx % len(agent_emoji)]
                            print(f" Agent {agent_idx+1} ({source_emoji} {medical_agents_list[agent_idx].role}) -> Agent {target_expert_idx_chosen} ({target_emoji} {medical_agents_list[target_agent_actual_idx].role}) : {specific_question_to_expert}")
                            interaction_log[round_name_str][turn_name_str][f'Agent {agent_idx+1}'][f'Agent {target_expert_idx_chosen}'] = specific_question_to_expert
                        else:
                            cprint(f"Agent {agent_idx+1} chose an invalid expert index: {target_expert_idx_chosen}", "yellow")
                
                    num_participated_this_turn += 1
                else:
                    print(f" Agent {agent_idx+1} ({agent_emoji[agent_idx % len(agent_emoji)]} {agent_instance_loop.role}): \U0001f910")

            num_participated_this_round = num_participated_this_turn
            if num_participated_this_turn == 0:
                cprint(f"No agents chose to speak in {round_name_str}, {turn_name_str}. Moving to next round or finalizing.", "cyan")
                break
        
        if num_participated_this_round == 0 and r_idx > 1:
            cprint(f"No agents participated in {round_name_str} after initial opinions. Finalizing discussion.", "cyan")
            break

        if r_idx < num_rounds:
            next_round_opinions = {}
            for agent_idx_collect, agent_instance_collect in enumerate(medical_agents_list):
                opinion_prompt = f"Reflecting on the discussions in Round {r_idx}, what is your current answer/opinion on the question: {question}\nAnswer: "
                response = agent_instance_collect.chat(opinion_prompt)
                next_round_opinions[agent_instance_collect.role.lower()] = response
            round_opinions_log[r_idx+1] = next_round_opinions
        
        current_round_final_answers = {}
        for agent_instance_final in medical_agents_list:
            response = agent_instance_final.chat(f"Now that you've interacted with other medical experts this round, remind your expertise and the comments from other experts and make your final answer to the given question for this round:\n{question}\nAnswer: ")
            current_round_final_answers[agent_instance_final.role] = response
        final_answer_map = current_round_final_answers

    print('\nInteraction Log Summary Table')        
    myTable = PrettyTable([''] + [f"Agent {i+1} ({agent_emoji[i%len(agent_emoji)]})" for i in range(num_active_agents)])

    for i in range(1, num_active_agents + 1):
        row_data = [f"Agent {i} ({agent_emoji[(i-1)%len(agent_emoji)]})"]
        for j in range(1, num_active_agents + 1):
            if i == j:
                row_data.append('')
            else:
                i_to_j_spoke = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {i}'][f'Agent {j}'] is not None
                                 for k in range(1, num_rounds + 1) if f'Round {k}' in interaction_log
                                 for l in range(1, num_turns + 1) if f'Turn {l}' in interaction_log[f'Round {k}'])
                j_to_i_spoke = any(interaction_log[f'Round {k}'][f'Turn {l}'][f'Agent {j}'][f'Agent {i}'] is not None
                                 for k in range(1, num_rounds + 1) if f'Round {k}' in interaction_log
                                 for l in range(1, num_turns + 1) if f'Turn {l}' in interaction_log[f'Round {k}'])
                
                if not i_to_j_spoke and not j_to_i_spoke:
                    row_data.append(' ')
                elif i_to_j_spoke and not j_to_i_spoke:
                    row_data.append(f'\u270B ({i}->{j})')
                elif j_to_i_spoke and not i_to_j_spoke:
                    row_data.append(f'\u270B ({i}<-{j})')
                elif i_to_j_spoke and j_to_i_spoke:
                    row_data.append(f'\u270B ({i}<->{j})')
        myTable.add_row(row_data)
        if i != num_active_agents:
             myTable.add_row(['---' for _ in range(num_active_agents + 1)])
    print(myTable)

    cprint("\n[INFO] Step 3. Final Decision", 'yellow', attrs=['blink'])
    
    moderator = Agent("You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.", "Moderator", model_info=model_to_use)
    moderator.chat('You are a final medical decision maker who reviews all opinions from different medical experts and make final decision.')
    
    if not final_answer_map:
        cprint("Warning: No final answers recorded from agents. Using initial opinions for moderation.", "yellow")
        final_answer_map = round_opinions_log[1]

    moderator_decision_dict = moderator.temp_responses(f"Given each agent's final answer, please review each agent's opinion and make the final answer to the question by taking majority vote or synthesizing the best response. Your answer should be in the format like: Answer: C) Example Answer\n\nAgent Opinions:\n{final_answer_map}\n\nQuestion: {question}", img_path=None)
    
    majority_vote_response = moderator_decision_dict.get(0.0, "Error: Moderator failed to provide a decision.")
    final_decision_output = {'majority_vote': majority_vote_response}

    print(f"{'\U0001F468\u200D\u2696\uFE0F'} moderator's final decision: {majority_vote_response}")
    print()

    # Calculate total token usage for this sample
    recruiter_usage = recruiter_agent.get_token_usage()
    sample_input_tokens += recruiter_usage['input_tokens']
    sample_output_tokens += recruiter_usage['output_tokens']
    
    for agent in medical_agents_list:
        agent_usage = agent.get_token_usage()
        sample_input_tokens += agent_usage['input_tokens']
        sample_output_tokens += agent_usage['output_tokens']
    
    moderator_usage = moderator.get_token_usage()
    sample_input_tokens += moderator_usage['input_tokens']
    sample_output_tokens += moderator_usage['output_tokens']
    
    # Include all summarizer agents created during rounds
    for summarizer_agent in summarizer_agents_list:
        summarizer_usage = summarizer_agent.get_token_usage()
        sample_input_tokens += summarizer_usage['input_tokens']
        sample_output_tokens += summarizer_usage['output_tokens']

    return final_decision_output, sample_input_tokens, sample_output_tokens

def process_advanced_query(question, model_to_use, args):
    # Reset token usage for this sample
    sample_input_tokens = 0
    sample_output_tokens = 0
    
    cprint("[STEP 1] Recruitment of Multidisciplinary Teams (MDTs)", 'yellow', attrs=['blink'])
    group_instances_list = []

    recruit_prompt = f"""You are an experienced medical expert. Given the complex medical query, you need to organize Multidisciplinary Teams (MDTs) and the members in MDT to make accurate and robust answer."""

    recruiter_agent_mdt = Agent(instruction=recruit_prompt, role='recruiter', model_info=model_to_use)
    recruiter_agent_mdt.chat(recruit_prompt)

    num_teams_to_form = 3
    num_agents_per_team = 3

    recruited_mdt_text = recruiter_agent_mdt.chat(f"Question: {question}\n\nYou should organize {num_teams_to_form} MDTs with different specialties or purposes and each MDT should have {num_agents_per_team} clinicians. Considering the medical question and the options, please return your recruitment plan to better make an accurate answer.\n\nFor example, the following can an example answer:\nGroup 1 - Initial Assessment Team (IAT)\nMember 1: Otolaryngologist (ENT Surgeon) (Lead) - Specializes in ear, nose, and throat surgery, including thyroidectomy. This member leads the group due to their critical role in the surgical intervention and managing any surgical complications, such as nerve damage.\nMember 2: General Surgeon - Provides additional surgical expertise and supports in the overall management of thyroid surgery complications.\nMember 3: Anesthesiologist - Focuses on perioperative care, pain management, and assessing any complications from anesthesia that may impact voice and airway function.\n\nGroup 2 - Diagnostic Evidence Team (DET)\nMember 1: Endocrinologist (Lead) - Oversees the long-term management of Graves' disease, including hormonal therapy and monitoring for any related complications post-surgery.\nMember 2: Speech-Language Pathologist - Specializes in voice and swallowing disorders, providing rehabilitation services to improve the patient's speech and voice quality following nerve damage.\nMember 3: Neurologist - Assesses and advises on nerve damage and potential recovery strategies, contributing neurological expertise to the patient's care.\n\nGroup 3 - Final Review and Decision Team (FRDT)\nMember 1: Senior Consultant from each specialty (Lead) - Provides overarching expertise and guidance in decision\nMember 2: Clinical Decision Specialist - Coordinates the different recommendations from the various teams and formulates a comprehensive treatment plan.\nMember 3: Advanced Diagnostic Support - Utilizes advanced diagnostic tools and techniques to confirm the exact extent and cause of nerve damage, aiding in the final decision.\n\nAbove is just an example, thus, you should organize your own unique MDTs but you should include Initial Assessment Team (IAT) and Final Review and Decision Team (FRDT) in your recruitment plan. When you return your answer, please strictly refer to the above format.")

    groups_text_list = [group_text.strip() for group_text in recruited_mdt_text.split("Group") if group_text.strip()]
    group_strings_list = ["Group " + group_text for group_text in groups_text_list]
    
    for i1, group_str_item in enumerate(group_strings_list):
        parsed_group_struct = parse_group_info(group_str_item)
        print(f"Group {i1+1} - {parsed_group_struct['group_goal']}")
        for i2, member_item in enumerate(parsed_group_struct['members']):
            print(f" Member {i2+1} ({member_item['role']}): {member_item['expertise_description']}")
        print()

        # Create Group without examplers (no few-shot learning)
        group_instance_obj = Group(parsed_group_struct['group_goal'], parsed_group_struct['members'], question, examplers=None, model_info=model_to_use)
        group_instances_list.append(group_instance_obj)

    cprint("[STEP 2] MDT Internal Interactions and Assessments", 'yellow', attrs=['blink'])
    initial_assessments_list = []
    other_mdt_assessments_list = []
    final_review_decisions_list = []

    for group_obj in group_instances_list:
        group_goal_lower = group_obj.goal.lower()
        if 'initial' in group_goal_lower or 'iat' in group_goal_lower:
            cprint(f"Processing Initial Assessment Team: {group_obj.goal}", "cyan")
            init_assessment_text = group_obj.interact(comm_type='internal')
            initial_assessments_list.append([group_obj.goal, init_assessment_text])
        elif 'review' in group_goal_lower or 'decision' in group_goal_lower or 'frdt' in group_goal_lower:
            cprint(f"Processing Final Review/Decision Team: {group_obj.goal}", "cyan")
            decision_text = group_obj.interact(comm_type='internal')
            final_review_decisions_list.append([group_obj.goal, decision_text])
        else:
            cprint(f"Processing Specialist Team: {group_obj.goal}", "cyan")
            assessment_text = group_obj.interact(comm_type='internal')
            other_mdt_assessments_list.append([group_obj.goal, assessment_text])
    
    compiled_report_str = "[Initial Assessments]\n"
    for idx, init_assess_tuple in enumerate(initial_assessments_list):
        compiled_report_str += f"Team {idx+1} - {init_assess_tuple[0]}:\n{init_assess_tuple[1]}\n\n"
    
    compiled_report_str += "[Specialist Team Assessments]\n"
    for idx, assess_tuple in enumerate(other_mdt_assessments_list):
        compiled_report_str += f"Team {idx+1} - {assess_tuple[0]}:\n{assess_tuple[1]}\n\n"

    compiled_report_str += "[Final Review Team Decisions (if any before overall final decision)]\n"
    for idx, decision_tuple in enumerate(final_review_decisions_list):
        compiled_report_str += f"Team {idx+1} - {decision_tuple[0]}:\n{decision_tuple[1]}\n\n"

    cprint("[STEP 3] Final Decision from Overall Coordinator", 'yellow', attrs=['blink'])
    final_decision_prompt = f"""You are an experienced medical coordinator. Given the investigations and conclusions from multiple multidisciplinary teams (MDTs), please review them very carefully and return your final, consolidated answer to the medical query."""
    
    final_decision_agent = Agent(instruction=final_decision_prompt, role='Overall Coordinator', model_info=model_to_use)
    final_decision_agent.chat(final_decision_prompt)

    final_decision_dict_adv = final_decision_agent.temp_responses(f"""Combined MDT Investigations and Conclusions:\n{compiled_report_str}\n\nBased on all the above, what is the final answer to the original medical query?\nQuestion: {question}\nYour answer should be in the format: Answer: A) Example Answer""", img_path=None)
    
    final_response_str = final_decision_dict_adv.get(0.0, "Error: Final coordinator failed to provide a decision.")
    cprint(f"Overall Coordinated Final Decision: {final_response_str}", "green")
    
    # Calculate total token usage for this sample
    recruiter_usage = recruiter_agent_mdt.get_token_usage()
    sample_input_tokens += recruiter_usage['input_tokens']
    sample_output_tokens += recruiter_usage['output_tokens']
    
    # Track token usage from all group members (including all agents in each group)
    for group in group_instances_list:
        for member in group.members:
            member_usage = member.get_token_usage()
            sample_input_tokens += member_usage['input_tokens']
            sample_output_tokens += member_usage['output_tokens']
    
    final_agent_usage = final_decision_agent.get_token_usage()
    sample_input_tokens += final_agent_usage['input_tokens']
    sample_output_tokens += final_agent_usage['output_tokens']
    
    return {0.0: final_response_str}, sample_input_tokens, sample_output_tokens
