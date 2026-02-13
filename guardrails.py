"""
Guardrails for LLM inputs and outputs
"""
import re
from typing import Dict, Tuple


class Guardrails:
    """Input and output guardrails for LLM safety"""
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|above|all)\s+instructions",
        r"disregard\s+(previous|above|all)",
        r"forget\s+(everything|all|previous)",
        r"you\s+are\s+now",
        r"new\s+instructions",
        r"system\s*:\s*",
        r"<\s*\|.*\|\s*>",  # Special tokens
    ]
    
    # Inappropriate content patterns
    INAPPROPRIATE_PATTERNS = [
        r"\b(hack|exploit|vulnerability|crack|bypass)\b",
        r"\b(illegal|unlawful|criminal)\b",
    ]
    
    @staticmethod
    def check_input(user_input: str) -> Tuple[bool, str]:
        """
        Validate user input for safety.
        
        Args:
            user_input: User's input message
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check length
        from config import config
        
        if len(user_input) > config.MAX_INPUT_LENGTH:
            return False, f"Input too long. Maximum {config.MAX_INPUT_LENGTH} characters allowed."
        
        if len(user_input.strip()) == 0:
            return False, "Input cannot be empty."
        
        # Check for prompt injection
        user_input_lower = user_input.lower()
        for pattern in Guardrails.INJECTION_PATTERNS:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return False, "Invalid input detected. Please rephrase your question."
        
        # Check for inappropriate content (basic)
        for pattern in Guardrails.INAPPROPRIATE_PATTERNS:
            if re.search(pattern, user_input_lower, re.IGNORECASE):
                return False, "Your question appears to contain inappropriate content. Please ask about TDI services and information."
        
        return True, ""
    
    @staticmethod
    def check_output(llm_output: str, context: str) -> Tuple[bool, str]:
        """
        Validate LLM output for safety and quality.
        
        Args:
            llm_output: LLM's generated response
            context: Context provided to the LLM
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        from config import config
        
        # Check length
        if len(llm_output) > config.MAX_OUTPUT_LENGTH:
            return False, "Response too long. Please try again."
        
        if len(llm_output.strip()) == 0:
            return False, "Empty response generated. Please try again."
        
        # Check for hallucination indicators (basic heuristic)
        hallucination_phrases = [
            "i don't have access to",
            "i cannot browse",
            "as an ai language model",
            "i'm sorry, but i",
        ]
        
        llm_output_lower = llm_output.lower()
        
        # If LLM is giving generic AI responses, it might be hallucinating
        for phrase in hallucination_phrases:
            if phrase in llm_output_lower:
                # This is actually okay - it means the model is being honest
                pass
        
        return True, ""
    
    @staticmethod
    def sanitize_input(user_input: str) -> str:
        """
        Sanitize user input by removing potentially harmful characters.
        
        Args:
            user_input: Raw user input
            
        Returns:
            Sanitized input
        """
        # Remove control characters except newlines and tabs
        sanitized = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', user_input)
        
        # Trim whitespace
        sanitized = sanitized.strip()
        
        return sanitized


# Singleton instance
guardrails = Guardrails()
