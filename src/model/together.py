import os
import random
import time
from datetime import timedelta
from typing import Any, Dict, Optional

from together import Together

from src import cache
from src.model.model import Model


class TogetherModel(Model):
    """Together API wrapper for chat/completion inference."""

    def __init__(
        self,
        engine: str = "meta-llama/Llama-3.1-8B-Instruct-Turbo",
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
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

        self.provider = provider
        self.base_url = base_url
        self.api_key_env = api_key_env

        self.gpt_waittime = 60
        self.client: Optional[Together] = None

    def _resolve_api_key(self) -> Optional[str]:
        env_name = self.api_key_env
        if not env_name:
            env_name = "TOGETHER_API_KEY"
        return os.getenv(env_name)

    def _get_client(self) -> Together:
        if self.client:
            return self.client

        api_key = self._resolve_api_key()
        if not api_key:
            raise RuntimeError("Missing Together API key. Set TOGETHER_API_KEY or pass api_key_env.")

        if self.base_url:
            try:
                self.client = Together(api_key=api_key, base_url=self.base_url)
            except TypeError:
                self.client = Together(api_key=api_key)
        else:
            self.client = Together(api_key=api_key)

        return self.client

    def _extract_usage_tokens(self, raw: Any) -> (int, int):
        usage = raw.get("usage") if isinstance(raw, dict) else getattr(raw, "usage", None)
        if usage is None:
            return 0, 0

        if isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
        else:
            prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
            completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))

        return int(prompt_tokens or 0), int(completion_tokens or 0)

    def __update_cost__(self, raw: Any):
        prompt_tokens, completion_tokens = self._extract_usage_tokens(raw)

        self.last_prompt_tokens = prompt_tokens
        self.last_completion_tokens = completion_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        if not (self.prompt_cost is not None and self.completion_cost is not None):
            return

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

        if self.api_endpoint == "completion":
            out = self.__safe_completion_call__(prompt, *args, **kwargs)
        else:
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
    ) -> Any:
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

        model_name = self.engine
        client = self._get_client()
        last_exc = None

        for _ in range(self.api_max_attempts):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n": num_samples,
                }
                if stop_token:
                    request_kwargs["stop"] = stop_token

                return client.chat.completions.create(**request_kwargs)
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

    def __safe_completion_call__(
        self,
        prompt: str,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        stop_token: str = None,
        num_samples: int = None,
    ) -> Any:
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

        model_name = self.engine
        client = self._get_client()
        last_exc = None

        for _ in range(self.api_max_attempts):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_tokens": max_tokens,
                    "n": num_samples,
                }
                if stop_token:
                    request_kwargs["stop"] = stop_token

                return client.completions.create(**request_kwargs)
            except Exception as e:
                last_exc = e
                print(f"ERROR: MODEL API Error: {e}")
                time.sleep(self.gpt_waittime + int(random.randint(1, 10)))

        return {
            "choices": [
                {
                    "text": f"{prompt} MODEL API Error - {last_exc}",
                }
            ],
            "API Error": True,
        }
