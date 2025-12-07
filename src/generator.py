from typing import List, Dict, Union
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.prompt import TNG_PROMPT
from src.guardrails import enforce_grounding

TNGD_FAQ_URL = "https://support.tngdigital.com.my/hc/en-my/categories/360002280493-Frequently-Asked-Questions-FAQ"


def _extract_answer(text: str) -> str:
    if not isinstance(text, str):
        return ""
    if "A:" in text:
        return text.split("A:", 1)[1].strip()
    if "a:" in text and "Q:" not in text:
        return text.split("a:", 1)[1].strip()
    return text.strip()


class TransformerGenerator:
    def __init__(self, model_name: str = "google/flan-t5-small", device: str = "cpu"):
        self.device = torch.device(device)
        self._tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self._model.eval()
        self.max_input_len = int(min(getattr(self._tok, "model_max_length", 1024), 1024))

        # diagnostics
        self.last_prompt_length = None
        self.last_model_raw_initial = None
        self.last_model_raw_retry = None

    def _build_prompt(self, question: str, retrieved: List[Dict], top_k: int = 5) -> str:
        parts = []
        for i, r in enumerate(retrieved[:top_k], start=1):
            url = r.get("url", "") or ""
            fused = float(r.get("fused_score", r.get("score", 0.0)) or 0.0)
            txt = _extract_answer(r.get("chunk_text", "") or "")
            header = f"Source {i} | fused_score={fused:.4f} | url: {url}"
            parts.append(f"{header}\n{txt}")
        sources = "\n\n".join(parts)
        return TNG_PROMPT.format(sources=sources, question=question, faq_url=TNGD_FAQ_URL)

    def generate(
        self,
        question: str,
        retrieved: List[Dict],
        max_tokens: int = 180,
        top_k_chunks: int = 5,
        temperature: float = 0.0,
        return_metadata: bool = False,
    ) -> Union[str, Dict]:
        """
        Generate an answer string from the model.

        If return_metadata=True, returns a dict:
          {
            "answer": <string>,
            "initial_raw": <raw initial output string>,
            "retry_raw": <raw retry output string or None>,
            "prompt_length": <tokens>,
            "prompt_full": <prompt text>
          }
        """
        # build prompt (we keep it here so generator can report it in metadata)
        prompt = self._build_prompt(question, retrieved, top_k=top_k_chunks)

        # prompt token diagnostics
        try:
            prompt_token_count = len(self._tok.encode(prompt, truncation=False))
        except Exception:
            tmp = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_len, padding="longest")
            prompt_token_count = int(tmp["input_ids"].shape[1])
        self.last_prompt_length = int(prompt_token_count)

        # encode (truncated) for model
        encoded = self._tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input_len, padding="longest").to(self.device)
        gen_kwargs = {"input_ids": encoded["input_ids"], "attention_mask": encoded["attention_mask"], "max_new_tokens": max_tokens}

        # decoding strategy
        try:
            temp_val = float(temperature)
        except Exception:
            temp_val = 0.0
        if temp_val == 0.0:
            gen_kwargs.update(num_beams=6, early_stopping=True, do_sample=False, no_repeat_ngram_size=3)
        else:
            gen_kwargs.update(do_sample=True, temperature=temp_val, top_p=0.95, top_k=50, no_repeat_ngram_size=3)

        def _gen_once(kwargs):
            try:
                with torch.no_grad():
                    dec_start = getattr(self._model.config, "decoder_start_token_id", None)
                    if dec_start is not None:
                        kwargs["decoder_start_token_id"] = dec_start
                    out_ids = self._model.generate(**kwargs)
                return self._tok.decode(out_ids[0], skip_special_tokens=True).strip()
            except Exception:
                return ""

        # initial generation
        initial = _gen_once(gen_kwargs)
        self.last_model_raw_initial = initial

        # quick echo test
        def _looks_like_echo(output: str, prompt_text: str) -> bool:
            # existing sanity checks
            if not output:
                return True
            o = " ".join(output.split()).lower()
            p = " ".join(prompt_text.split()).lower()

            # very short/single token responses are suspicious
            if len(o.split()) == 1:
                return True

            # exact substring copy of prompt (fast path)
            if o in p:
                return True

            # prompt contains the output (other direction)
            if p in o:
                return True

            # explicit rule: if output contains obvious instruction lines, treat as echo
            instr_markers = [
                "you are the official tng ewallet faq assistant",
                "use only the provided sources",
                "critical rules",
                "rule a:",
                "rule b:",
            ]
            for m in instr_markers:
                if m in o:
                    return True

            # token overlap heuristic (more robust)
            try:
                import re
                o_tokens = {t for t in re.findall(r"\w+", o) if len(t) > 2}
                p_tokens = {t for t in re.findall(r"\w+", p) if len(t) > 2}
                if len(o_tokens) > 0 and len(p_tokens) > 0:
                    overlap = len(o_tokens & p_tokens) / max(1, len(o_tokens))
                    # if more than half of output tokens appear in the prompt, consider echo
                    if overlap >= 0.5:
                        return True
            except Exception:
                pass

            # some canned rejection phrases from model should also be considered echo
            if "if the question" in o or "request blocked" in o or "this topic is not in our curated" in o:
                return True

            return False


        candidate = initial
        if _looks_like_echo(candidate, prompt):
            # sampled retries with increasing temp
            for temp, extra in ((0.6, 80), (0.8, 120), (1.0, 160)):
                retry_kwargs = gen_kwargs.copy()
                retry_kwargs.update({"do_sample": True, "temperature": temp, "max_new_tokens": max_tokens + extra})
                out = _gen_once(retry_kwargs)
                self.last_model_raw_retry = out
                if out and not _looks_like_echo(out, prompt) and len(out.split()) >= 4:
                    candidate = out
                    break
            else:
                candidate = ""  # force fallback

        # short / no valid candidate -> fallback
        if not candidate or len(candidate.split()) < 4:
            fallback = f"This topic is not in our curated TNG eWallet knowledge base. Please check: {TNGD_FAQ_URL}"
            if return_metadata:
                return {
                    "answer": fallback,
                    "initial_raw": self.last_model_raw_initial,
                    "retry_raw": self.last_model_raw_retry,
                    "prompt_length": self.last_prompt_length,
                    "prompt_full": prompt,
                }
            return fallback

        # strip Sources: footer (we still return full candidate in metadata)
        answer_part = candidate
        if "Sources:" in candidate:
            idx = candidate.rfind("Sources:")
            answer_part = candidate[:idx].strip()

        # Note: generator no longer enforces grounding here. rag_bot.py is authoritative.

        if return_metadata:
            return {
                "answer": candidate,
                "initial_raw": self.last_model_raw_initial,
                "retry_raw": self.last_model_raw_retry,
                "prompt_length": self.last_prompt_length,
                "prompt_full": prompt,
            }

        return candidate