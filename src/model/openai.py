import os
import random
import time
from datetime import timedelta
from typing import Any, Dict, Optional

from litellm import completion

from src import cache
from src.model.model import Model


class OpenAIModel(Model):
    """
    Provider-agnostic chat wrapper built on LiteLLM.

    Despite the legacy class name, this can target OpenAI and OpenAI-compatible APIs (e.g. Together)
    by setting provider/base_url/api_key_env.
    """

    def __init__(
        self,
        engine: str = "gpt-4o-mini",
        api_max_attempts: int = 30,
        api_endpoint: str = "chat",
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop_token: str = None,
        log_probs: int = 1,
        num_samples: int = 1,
        echo: bool = False,
        prompt_cost: float = None,
        completion_cost: float = None,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
    ):
        self.engine = engine
        self.api_max_attempts = api_max_attempts
        self.api_endpoint = api_endpoint.lower()

        self.max_tokens = max_tokens
        self.stop_token = stop_token
        self.log_probs = log_probs
        self.num_samples = num_samples
        self.echo = echo
        self.temperature = temperature
        self.top_p = top_p

        self.prompt_cost = prompt_cost
        self.completion_cost = completion_cost
        self.total_cost = 0.0

        self.provider = provider
        self.base_url = base_url
        self.api_key_env = api_key_env

        self.gpt_waittime = 60

    def _resolve_api_key(self) -> Optional[str]:
        env_name = self.api_key_env
        if not env_name:
            if self.provider == "together":
                env_name = "TOGETHER_API_KEY"
            else:
                env_name = "OPENAI_API_KEY"
        return os.getenv(env_name)

    def _resolved_model_name(self) -> str:
        if self.provider and "/" not in self.engine:
            return f"{self.provider}/{self.engine}"
        return self.engine

    def __update_cost__(self, raw: Any):
        if not (self.prompt_cost and self.completion_cost):
            return

        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        if usage is None:
            return

        prompt_tokens = usage.get("prompt_tokens", 0) if isinstance(usage, dict) else getattr(usage, "prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0) if isinstance(usage, dict) else getattr(usage, "completion_tokens", 0)

        cost = completion_tokens * self.completion_cost + prompt_tokens * self.prompt_cost
        self.total_cost += cost

    @cache.cached(
        data_ex=timedelta(days=30),
        no_data_ex=timedelta(hours=1),
        prepended_key_attr="engine,provider,base_url,num_samples,temperature=float(0),top_p=float(1.0),stop_token,max_tokens",
    )
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        if self.api_endpoint not in {"chat", "completion"}:
            raise Exception(f"Unknown api endpoint for model: {self.api_endpoint}")

        out = self.__safe_chat_call__(prompt, *args, **kwargs)
        self.__update_cost__(out)
        return out

    def __safe_chat_call__(
        self,
        prompt: str,
        system_prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop_token: str = None,
        num_samples: int = None,
    ) -> Dict[str, Any]:
        if max_tokens is None:
            max_tokens = self.max_tokens
        if temperature is None:
            temperature = self.temperature
        if top_p is None:
            top_p = self.top_p
        if stop_token is None:
            stop_token = self.stop_token
        if num_samples is None:
            num_samples = self.num_samples

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]

        model_name = self._resolved_model_name()
        api_key = self._resolve_api_key()
        last_exc = None

        for _ in range(self.api_max_attempts):
            try:
                kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n": num_samples,
                }
                if stop_token:
                    kwargs["stop"] = stop_token
                if self.base_url:
                    kwargs["api_base"] = self.base_url
                if api_key:
                    kwargs["api_key"] = api_key

                return completion(**kwargs)
            except Exception as e:
                last_exc = e
                print(f"ERROR: MODEL API Error: {e}")
                time.sleep(self.gpt_waittime + int(random.randint(1, 10)))

        return {
            "choices": [
                {
                    "message": {
                        "content": f"{prompt} MODEL API Error - {last_exc}",
                    }
                }
            ],
            "API Error": True,
        }
