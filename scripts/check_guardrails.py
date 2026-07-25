import sys
import yaml

def check_guardrails(prompt_file):
    print(f"🔒 Checking Security Guardrails for: {prompt_file}")
    
    with open(prompt_file, 'r') as f:
        data = yaml.safe_load(f)
        
    prompt_text = data.get("system_prompt", "").lower()
    
    forbidden_terms = [
        "ignore previous instructions",
        "system override",
        "bypass security",
        "secret_token"
    ]
    
    for term in forbidden_terms:
        if term in prompt_text:
            print(f"❌ SECURITY GUARDRAIL FAILED: Found restricted phrase '{term}' in prompt!")
            sys.exit(1)  # Fails the pipeline
            
    print("✅ All Security Guardrail checks passed!")

if __name__ == "__main__":
    prompt_path = sys.argv[1] if len(sys.argv) > 1 else "prompts/prompt_v1.yaml"
    check_guardrails(prompt_path)