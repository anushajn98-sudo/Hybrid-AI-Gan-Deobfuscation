import os
import torch
import torch.nn as nn
import google.generativeai as genai
from flask import Flask, render_template, request, redirect, url_for, flash, session
from database import init_db
from models import User, History
from TRAIN import Seq2SeqLSTM
import ast
import subprocess
import sys
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import joblib
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import numpy as np
import tokenize
from io import StringIO
import nltk
import re
import json
import random

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

# Initialize database
init_db()

# Configure Gemini
genai.configure(api_key="AIzaSyDLqBNTJzVD9M0X5uAPBVZHWdH7eS_fvf0")
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# Download NLTK data for BLEU score calculation
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# ----------------------------
# GAN Discriminator Model
# ----------------------------
class Discriminator(nn.Module):
    def __init__(self, vocab_size, hidden=128, nclass=6, max_len=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.adv = nn.Linear(hidden, 1)
        self.cls = nn.Linear(hidden, nclass)

    def forward(self, x):
        emb = self.embed(x)
        _, (h, _) = self.lstm(emb)
        h = h[-1]
        validity = torch.sigmoid(self.adv(h))
        cls_logits = self.cls(h)
        return validity, cls_logits

# Load LSTM Model
def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = Seq2SeqLSTM(len(ckpt["vocab"]), hidden=256, nclass=len(ckpt["mtypes"]))
    model.load_state_dict(ckpt["model"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, ckpt["vocab"], ckpt["mtypes"], device

# Load CodeBERT Model
def load_codebert_model(model_path):
    try:
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(model_path)
        label_encoder = joblib.load(f"{model_path}/label_encoder.pkl")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, label_encoder, device
    except Exception as e:
        print(f"Error loading CodeBERT model: {e}")
        return None, None, None, None

# Load GAN Model
def load_gan_model(model_path):
    try:
        ckpt = torch.load(model_path, map_location="cpu")
        vocab, mtypes = ckpt["vocab"], ckpt["mtypes"]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        D = Discriminator(len(vocab), hidden=128, nclass=len(mtypes)).to(device)
        D.load_state_dict(ckpt["D"])
        D.eval()
        
        return D, vocab, mtypes, device
    except Exception as e:
        print(f"Error loading GAN model: {e}")
        return None, {}, ["benign", "malware"], "cpu"

# Load the models
try:
    lstm_model, vocab, mtypes, lstm_device = load_model("lstm_model.pt")
except:
    print("Warning: Could not load LSTM model. Using dummy model.")
    lstm_model, vocab, mtypes, lstm_device = None, {}, ["benign", "malware"], "cpu"

try:
    codebert_tokenizer, codebert_model, label_encoder, codebert_device = load_codebert_model("./malware_codebert_model")
except:
    print("Warning: Could not load CodeBERT model.")
    codebert_tokenizer, codebert_model, label_encoder, codebert_device = None, None, None, None

try:
    gan_model, gan_vocab, gan_mtypes, gan_device = load_gan_model("gan_classifier.pt")
except:
    print("Warning: Could not load GAN model.")
    gan_model, gan_vocab, gan_mtypes, gan_device = None, {}, ["benign", "malware"], "cpu"

# Load dataset for testing
def load_test_data(dataset_path="python_deobf_dataset.jsonl"):
    test_data = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        test_data.append(data)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON line: {line}")
                        continue
        return test_data
    except FileNotFoundError:
        print(f"Warning: Dataset file {dataset_path} not found")
        return []
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

# Get a random sample from dataset
def get_random_sample():
    test_data = load_test_data("python_deobf_dataset.jsonl")
    if test_data:
        return random.choice(test_data)
    return None

# Inference (LSTM)
def infer_lstm(model, vocab, mtypes, obf, device, max_len=256):
    if model is None:
        return "Model not loaded", "Unknown", {}
        
    ivocab = {i: c for c, i in vocab.items()}
    arr = [vocab.get(c, 3) for c in obf][:max_len-1] + [2]
    src = torch.tensor([arr + [0]*(max_len-len(arr))]).to(device)

    with torch.no_grad():
        out, cls = model(src)

    predseq = out.argmax(-1)[0].tolist()
    deobf = "".join(ivocab.get(x, "") for x in predseq if x > 3)
    labelid = cls.argmax(-1).item()
    malware_type = mtypes[labelid]
    
    # Get probabilities
    probabilities = torch.softmax(cls, dim=1)[0].cpu().numpy()
    confidence_scores = {mtypes[i]: float(probabilities[i]) for i in range(len(mtypes))}
    
    return deobf, malware_type, confidence_scores

# Inference (CodeBERT)
def infer_codebert(tokenizer, model, label_encoder, code_snippet, device):
    if model is None or tokenizer is None or label_encoder is None:
        return "Model not loaded", {}
    
    try:
        # Tokenize input
        inputs = tokenizer(code_snippet, return_tensors="pt", padding="max_length", 
                          truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Run model
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)[0].cpu().numpy()
            predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()

        # Decode label and get confidence scores
        predicted_label = label_encoder.inverse_transform(predictions)[0]
        confidence_scores = {label_encoder.classes_[i]: float(probabilities[i]) 
                           for i in range(len(label_encoder.classes_))}
        
        return predicted_label, confidence_scores
    except Exception as e:
        return f"Error: {str(e)}", {}

# Inference (GAN)
def infer_gan(model, vocab, mtypes, obf, device, max_len=256):
    if model is None:
        return "Model not loaded", "Unknown", {}
    
    # Encode input
    arr = [vocab.get(c, 3) for c in obf][:max_len]
    arr = arr + [0] * (max_len - len(arr))
    seq = torch.tensor([arr], device=device)

    with torch.no_grad():
        _, cls_logits = model(seq)
        probabilities = torch.softmax(cls_logits, dim=1)[0].cpu().numpy()
        pred = cls_logits.argmax(-1).item()
        malware_type = mtypes[pred]
        
        confidence_scores = {mtypes[i]: float(probabilities[i]) for i in range(len(mtypes))}
    
    return malware_type, confidence_scores

# Improved AST analysis with better error handling
def analyze_ast(code):
    try:
        if not code or not code.strip():
            return {'errors': ['No code to analyze']}
        
        # Clean the code first - remove any non-Python content
        clean_code = extract_python_code(code)
        if not clean_code:
            return {'errors': ['No valid Python code found for AST analysis']}
        
        # Parse the code into AST
        tree = ast.parse(clean_code)
        
        analysis = {
            'functions': [],
            'classes': [],
            'imports': [],
            'calls': [],
            'variables': set(),
            'errors': []
        }
        
        # Recursive function to analyze nodes
        def analyze_node(node):
            if isinstance(node, ast.FunctionDef):
                analysis['functions'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                analysis['classes'].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    analysis['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    analysis['imports'].append(f"{module}.{alias.name}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    analysis['calls'].append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    if isinstance(node.func.value, ast.Name):
                        analysis['calls'].append(f"{node.func.value.id}.{node.func.attr}")
                    else:
                        analysis['calls'].append(node.func.attr)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                analysis['variables'].add(node.id)
                
            # Recursively analyze child nodes
            for child in ast.iter_child_nodes(node):
                analyze_node(child)
        
        analyze_node(tree)
        analysis['variables'] = list(analysis['variables'])
        
        return analysis
    except SyntaxError as e:
        return {'errors': [f'Syntax error in AST parsing: {str(e)}']}
    except Exception as e:
        return {'errors': [f'AST parsing error: {str(e)}']}

# Function to extract only Python code from Gemini response
def extract_python_code(text):
    """Extract Python code from text, handling various formats"""
    if not text:
        return ""
    
    # Try to extract code from markdown code blocks
    code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', text, re.DOTALL)
    if code_blocks:
        return code_blocks[0].strip()
    
    # Try to extract between markers
    marker_match = re.search(r'DECODED_CODE_START\s*(.*?)\s*DECODED_CODE_END', text, re.DOTALL)
    if marker_match:
        return marker_match.group(1).strip()
    
    # If no markers found, try to find the first Python-like code block
    lines = text.split('\n')
    code_lines = []
    in_code = False
    
    for line in lines:
        # Look for lines that look like Python code
        if (line.strip().startswith(('import ', 'from ', 'def ', 'class ', 'print(', 'if ', 'for ', 'while ')) or
            '=' in line and not line.strip().startswith('#') and len(line.split('=')) == 2):
            in_code = True
        
        if in_code:
            # Stop if we hit something that doesn't look like code
            if line.strip() and not any(c in line for c in [' ', '\t', '=', '(', ')', '[', ']', '{', '}', ':']):
                if not line.strip().startswith(('#', '"', "'")):
                    break
            code_lines.append(line)
    
    result = '\n'.join(code_lines).strip()
    
    # Basic validation - check if it looks like Python code
    if any(keyword in result for keyword in ['import', 'def', 'class', 'print', '=']):
        return result
    
    return ""

# Calculate BLEU score
def calculate_bleu(reference, candidate):
    try:
        if not reference or not candidate:
            return 0.0
            
        # Tokenize the reference and candidate
        reference_tokens = [nltk.word_tokenize(reference.lower())]
        candidate_tokens = nltk.word_tokenize(candidate.lower())
        
        # Calculate BLEU score with smoothing
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu(reference_tokens, candidate_tokens, 
                                  smoothing_function=smoothie)
        
        return round(bleu_score * 100, 2)  # Convert to percentage
    except:
        return 0.0

# Improved Gemini inference with stricter formatting
def infer_gemini(obf):
    prompt = f"""DEOBFUSCATION TASK:
You are a Python deobfuscation tool. Given obfuscated Python code, return ONLY two things:

1. The clean, executable Python code
2. The expected output when this code runs

FORMAT REQUIREMENTS:
- Start with exactly: DECODED_CODE_START
- Then put the clean Python code
- Then exactly: DECODED_CODE_END
- Then exactly: EXPECTED_OUTPUT_START  
- Then put the expected output
- Then exactly: EXPECTED_OUTPUT_END

DO NOT:
- Add any explanations
- Use markdown formatting
- Add comments to the code
- Include any other text

OBFUSCATED CODE:
{obf}
"""
    
    try:
        chat = gemini_model.start_chat(history=[])
        gemini_response = chat.send_message(prompt)
        return gemini_response.text
    except Exception as e:
        return f"Error contacting Gemini API: {str(e)}"

# Much stricter Gemini response parsing
def parse_gemini_response(response):
    if response.startswith("Error"):
        return response, "Could not get response from Gemini"
    
    # Clean the response first - remove any markdown formatting
    clean_response = re.sub(r'\*\*|\*|`', '', response)  # Remove markdown
    
    # Strict extraction using markers
    code_pattern = r'DECODED_CODE_START\s*(.*?)\s*DECODED_CODE_END'
    output_pattern = r'EXPECTED_OUTPUT_START\s*(.*?)\s*EXPECTED_OUTPUT_END'
    
    code_match = re.search(code_pattern, clean_response, re.DOTALL)
    output_match = re.search(output_pattern, clean_response, re.DOTALL)
    
    if code_match and output_match:
        deobf_code = code_match.group(1).strip()
        expected_output = output_match.group(1).strip()
        return deobf_code, expected_output
    
    # If markers not found, try to extract Python code more aggressively
    deobf_code = extract_python_code(clean_response)
    expected_output = extract_expected_output(clean_response, deobf_code)
    
    return deobf_code, expected_output

def execute_python_code_safely(code, timeout=10):
    try:
        if not code or not code.strip():
            return {
                'success': False,
                'stdout': '',
                'stderr': 'No valid code to execute',
                'returncode': -1
            }
        
        # Use the simple but effective sanitization
        clean_code = sanitize_python_code(code)
        
        # Create a temporary file with unique name
        import tempfile
        import uuid
        
        temp_filename = f"temp_execution_{uuid.uuid4().hex[:8]}.py"
        
        with open(temp_filename, 'w', encoding='utf-8') as f:
            f.write(clean_code)
        
        # Execute the file
        result = subprocess.run(
            [sys.executable, temp_filename],
            capture_output=True, 
            text=True, 
            timeout=timeout,
            shell=False
        )
        
        # Clean up
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass
        
        return {
            'success': result.returncode == 0,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except subprocess.TimeoutExpired:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass
        return {
            'success': False,
            'stdout': '',
            'stderr': 'Execution timed out (10 seconds)',
            'returncode': -1
        }
    except Exception as e:
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except:
            pass
        return {
            'success': False,
            'stdout': '',
            'stderr': f'Execution error: {str(e)}',
            'returncode': -1
        }

def sanitize_python_code(code):
    """Simple but effective code sanitization"""
    # First, use the simple indentation fixer
    fixed_code = simple_indentation_fix(code)
    
    # Remove any non-Python lines
    lines = fixed_code.split('\n')
    python_lines = []
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
            
        # Keep only lines that look like Python code
        if (any(stripped.startswith(keyword) for keyword in 
               ['import', 'from', 'def', 'class', 'print', 'if', 'for', 'while', 'return', 'try']) or
            '=' in stripped or '(' in stripped or ')' in stripped):
            python_lines.append(line)
    
    clean_code = '\n'.join(python_lines)
    
    # Ensure we have valid Python syntax
    try:
        ast.parse(clean_code)
        return clean_code
    except SyntaxError:
        # If still invalid, wrap in a simple function
        indented_code = clean_code.replace('\n', '\n    ')
        return "def main():\n    " + indented_code + "\n\nif __name__ == '__main__':\n    main()"

def extract_expected_output(text, code):
    """Extract expected output from text, removing the code part"""
    if not text:
        return ""
    
    # Remove the code part to isolate output
    clean_text = text.replace(code, '').strip()
    
    # Remove any remaining code-like lines
    lines = clean_text.split('\n')
    output_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(keyword in line.lower() for keyword in ['output', 'result', 'prints', 'displays']):
            continue
        if not any(c in line for c in ['import', 'def', 'class', '=', '(', ')', '[', ']']):
            output_lines.append(line)
    
    return '\n'.join(output_lines).strip()

# Alternative: Simple but effective indentation fixer
def simple_indentation_fix(code):
    """Simple approach to fix indentation - add 4 spaces to all lines after def/class"""
    lines = code.split('\n')
    fixed_lines = []
    in_block = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped:
            fixed_lines.append('')
            continue
            
        if stripped.startswith(('def ', 'class ')):
            fixed_lines.append(stripped)
            in_block = True
        elif in_block:
            # Add 4 spaces for everything inside the block
            fixed_lines.append('    ' + stripped)
        else:
            fixed_lines.append(stripped)
    
    return '\n'.join(fixed_lines)

# Lexical analysis
def analyze_lexical(code):
    try:
        if not code or not code.strip():
            return [{'error': 'No code to analyze'}]
        
        tokens = []
        for tok in tokenize.generate_tokens(StringIO(code).readline):
            tokens.append({
                'type': tokenize.tok_name[tok.type],
                'string': tok.string,
                'start': tok.start,
                'end': tok.end,
                'line': tok.line
            })
        return tokens
    except Exception as e:
        return [{'error': f'Lexical analysis error: {str(e)}'}]

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']
        
        if User.find_by_username(username):
            flash('Username already exists. Please choose a different one.', 'error')
            return redirect(url_for('register'))
        
        if User.create(username, password, email):
            flash('Registration successful. Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Registration failed. Please try again.', 'error')
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.find_by_username(username)
        if user and user['password'] == password:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password.', 'error')
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))

@app.route('/home')
def home():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    return render_template('home.html')

@app.route('/analyze_random')
def analyze_random():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    # Get a random sample from dataset
    sample = get_random_sample()
    if not sample:
        flash('No samples found in dataset.', 'error')
        return redirect(url_for('home'))
    
    user_input = sample.get('obfuscated', '')
    expected_deobf = sample.get('deobfuscated', '')
    expected_label = sample.get('label', 'unknown')
    
    # 1) LSTM inference
    deobf_lstm, malware_type_lstm, confidence_lstm = infer_lstm(
        lstm_model, vocab, mtypes, user_input, lstm_device
    )
    
    # 2) CodeBERT inference
    malware_type_codebert, confidence_codebert = infer_codebert(
        codebert_tokenizer, codebert_model, label_encoder, user_input, codebert_device
    )
    
    # 3) GAN inference
    malware_type_gan, confidence_gan = infer_gan(
        gan_model, gan_vocab, gan_mtypes, user_input, gan_device
    )
    
    # 4) Gemini inference for deobfuscation
    gemini_response = infer_gemini(user_input)
    deobf_code, gemini_output = parse_gemini_response(gemini_response)
    
    # Clean up Gemini output
    if '[Current date and time]' in deobf_code:
        deobf_code = deobf_code.replace('[Current date and time]', '').strip()
    if '[Current date and time]' in gemini_output:
        gemini_output = gemini_output.replace('[Current date and time]', '').strip()
    
    # Calculate BLEU score between LSTM and Gemini deobfuscations
    bleu_score = calculate_bleu(deobf_lstm, deobf_code)
    
    # Calculate BLEU score between expected and Gemini deobfuscations
    bleu_score_expected = calculate_bleu(expected_deobf, deobf_code)
    
    # Execute the deobfuscated code
    execution_result = execute_python_code_safely(deobf_code)
    
    # Perform AST and lexical analysis
    ast_analysis = analyze_ast(deobf_code)
    lexical_analysis = analyze_lexical(deobf_code)
    
    # Save to history
    History.add_record(
        session['user_id'], 
        user_input, 
        f"Deobfuscated Code:\n{deobf_code}\n\nOutput:\n{gemini_output}", 
        f"LSTM: {malware_type_lstm}, CodeBERT: {malware_type_codebert}, GAN: {malware_type_gan}"
    )
    
    return render_template('result.html', 
                          input_code=user_input,
                          deobf_code=deobf_code,
                          gemini_output=gemini_output,
                          malware_type_lstm=malware_type_lstm,
                          malware_type_codebert=malware_type_codebert,
                          malware_type_gan=malware_type_gan,
                          confidence_lstm=confidence_lstm,
                          confidence_codebert=confidence_codebert,
                          confidence_gan=confidence_gan,
                          bleu_score=bleu_score,
                          bleu_score_expected=bleu_score_expected,
                          execution_result=execution_result,
                          ast_analysis=ast_analysis,
                          lexical_analysis=lexical_analysis,
                          expected_deobf=expected_deobf,
                          expected_label=expected_label,
                          sample_id=sample.get('id', 'unknown'))

@app.route('/test_dataset')
def test_dataset():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    # Load test data
    test_data = load_test_data("python_deobf_dataset.jsonl")
    
    if not test_data:
        flash('No test data found. Please check if python_deobf_dataset.jsonl exists.', 'error')
        return redirect(url_for('home'))
    
    results = []
    for i, item in enumerate(test_data[:50]):  # Test first 50 items
        code = item.get('obfuscated', '')
        expected_label = item.get('label', 'unknown')
        item_id = item.get('id', f'item_{i+1}')
        
        # Run all models
        _, malware_type_lstm, confidence_lstm = infer_lstm(
            lstm_model, vocab, mtypes, code, lstm_device
        )
        
        malware_type_codebert, confidence_codebert = infer_codebert(
            codebert_tokenizer, codebert_model, label_encoder, code, codebert_device
        )
        
        malware_type_gan, confidence_gan = infer_gan(
            gan_model, gan_vocab, gan_mtypes, code, gan_device
        )
        
        results.append({
            'id': item_id,
            'code_preview': code[:100] + '...' if len(code) > 100 else code,
            'expected_label': expected_label,
            'lstm_result': malware_type_lstm,
            'codebert_result': malware_type_codebert,
            'gan_result': malware_type_gan,
            'lstm_correct': malware_type_lstm.lower() == expected_label.lower(),
            'codebert_correct': malware_type_codebert.lower() == expected_label.lower(),
            'gan_correct': malware_type_gan.lower() == expected_label.lower(),
            'lstm_confidence': confidence_lstm.get(malware_type_lstm, 0) * 100,
            'codebert_confidence': confidence_codebert.get(malware_type_codebert, 0) * 100,
            'gan_confidence': confidence_gan.get(malware_type_gan, 0) * 100
        })
    
    # Calculate accuracy
    if results:
        lstm_accuracy = sum(1 for r in results if r['lstm_correct']) / len(results) * 100
        codebert_accuracy = sum(1 for r in results if r['codebert_correct']) / len(results) * 100
        gan_accuracy = sum(1 for r in results if r['gan_correct']) / len(results) * 100
    else:
        lstm_accuracy = codebert_accuracy = gan_accuracy = 0
    
    return render_template('dataset_test.html',
                         results=results,
                         lstm_accuracy=lstm_accuracy,
                         codebert_accuracy=codebert_accuracy,
                         gan_accuracy=gan_accuracy)

@app.route('/history')
def history():
    if 'user_id' not in session:
        flash('Please log in to access this page.', 'error')
        return redirect(url_for('login'))
    
    history = History.get_user_history(session['user_id'])
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)