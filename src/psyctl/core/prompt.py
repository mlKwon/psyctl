import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable PyTorch compiler to avoid Triton issues
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.disable = True


class P2:
    """
    P2 is a prompt that is used to generate a personality-specific prompt.
    https://arxiv.org/abs/2206.07550
    """

    def __init__(self, model: AutoModelForCausalLM, tokenizer: AutoTokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.keywords = None
        self.personality = None
        self.keywords_build_prompt = None
        self.personality_build_prompt = None
        self.char_name = None

    def build(self, char_name: str, personality_trait: str):
        keywords_build_prompt, keywords = self._get_result(
            f"Words related to {personality_trait}? (format: Comma sperated words)"
        )
        personality_build_prompt, personality = self._get_result(
            f"{keywords} are traits of {char_name}.\n\nDesribe about {char_name}",
            prefill=f"Here's a description of {char_name}, built from the traits suggested by the list:",
        )
        self.char_name = char_name
        self.keywords = keywords
        self.personality = personality
        self.keywords_build_prompt = keywords_build_prompt
        self.personality_build_prompt = personality_build_prompt
        return self.personality

    def _get_result(self, prompt, prefill=None):
        messages = [{"role": "user", "content": prompt}]

        # 1. 유저 메시지를 chat template로 변환 (<|assistant|> 까지 포함)
        try:
            # Try with return_dict=True to get dictionary format
            tokenized_input = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,  # Get string first
                add_generation_prompt=True,
                return_tensors=None,
            )

            # Now tokenize the string
            tokenized = self.tokenizer(
                tokenized_input,
                return_tensors="pt",
                add_special_tokens=False,  # Chat template already adds special tokens
            )

        except Exception:
            # Fallback: some models don't support chat templates
            tokenized = self.tokenizer(
                prompt, return_tensors="pt", add_special_tokens=True
            )
        
        # 2. prefill이 있다면 assistant 답변의 시작 부분으로 추가
        if prefill:
            prefill_ids = self.tokenizer.encode(
                prefill, add_special_tokens=False, return_tensors="pt"
            )

            # tokenized["input_ids"]와 prefill_ids 이어붙이기
            tokenized["input_ids"] = torch.cat(
                [tokenized["input_ids"], prefill_ids], dim=1
            )

            # attention_mask도 확장
            prefill_attention = torch.ones_like(prefill_ids)
            tokenized["attention_mask"] = torch.cat(
                [tokenized["attention_mask"], prefill_attention], dim=1
            )

        # Move tensors to the same device as the model
        device = next(self.model.parameters()).device
        tokenized["input_ids"] = tokenized["input_ids"].to(device)
        tokenized["attention_mask"] = tokenized["attention_mask"].to(device)

        # 3. 모델 생성
        outputs = self.model.generate(
            input_ids=tokenized["input_ids"],
            attention_mask=tokenized["attention_mask"],
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        len_input = tokenized["input_ids"][0].shape[0]
        input_text = self.tokenizer.decode(
            tokenized["input_ids"][0], skip_special_tokens=True
        )
        output_text = self.tokenizer.decode(
            outputs[0, len_input:], skip_special_tokens=True
        )
        return input_text, output_text
