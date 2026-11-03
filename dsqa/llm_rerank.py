"""LLM làm reranker (pointwise): điểm = logit("Yes") − logit("No") ở token đầu câu trả lời.
Dùng lại model sinh đã load (Olmo) → không thêm dung lượng zip."""
import torch

SYSTEM = "You judge whether a passage from a document corpus is useful for answering a question."
PROMPT = ("Question: {q}\n\nPassage:\n{p}\n\n"
          "Does this passage contain information that helps answer the question? Answer Yes or No.")


class LLMReranker:
    def __init__(self, model, tokenizer, max_passage_tokens=400, batch_size=8):
        self.model, self.tok = model, tokenizer
        self.max_p, self.bs = max_passage_tokens, batch_size
        self.yes = tokenizer.encode("Yes", add_special_tokens=False)[0]
        self.no = tokenizer.encode("No", add_special_tokens=False)[0]

    def _prompt(self, question, passage):
        ids = self.tok.encode(passage, add_special_tokens=False)[: self.max_p]
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": PROMPT.format(q=question, p=self.tok.decode(ids))}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def __call__(self, question, passages):
        texts = [self._prompt(question, p) for p in passages]
        scores = []
        side, self.tok.padding_side = self.tok.padding_side, "left"  # token cuối của mọi dòng thẳng hàng
        try:
            for i in range(0, len(texts), self.bs):
                enc = self.tok(texts[i:i + self.bs], return_tensors="pt", padding=True,
                               add_special_tokens=False).to(self.model.device)
                logits = self.model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                scores += (logits[:, self.yes] - logits[:, self.no]).tolist()
        finally:
            self.tok.padding_side = side
        return scores
