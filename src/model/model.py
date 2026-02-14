from abc import abstractmethod, ABCMeta
from typing import Any


class Model(metaclass=ABCMeta):

    @abstractmethod
    def inference(self, prompt: str, *args, **kwargs) -> Any:
        """
        Most simple inference to a model that takes a prompt and gets an output object back.

        :param prompt: The prompt to give to the language model
        :param args: If the specific model needs more arguments
        :param kwargs: If the specific model needs more keyword arguments
        :return: Generated response from the language model.
        """
        raise NotImplementedError("All models need an inference call implemented.")


def extract_text_from_response(raw: Any) -> str:
    """
    Normalize text extraction across SDK response types.

    Supports string outputs, dict-like responses, OpenAI-style objects, and LiteLLM/OpenAI v1 chat payloads.
    """
    if raw is None:
        return ""

    if isinstance(raw, str):
        return raw

    try:
        choices = raw.get("choices") if isinstance(raw, dict) else getattr(raw, "choices", None)
        if choices:
            first_choice = choices[0]

            if isinstance(first_choice, dict):
                message = first_choice.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if content is not None:
                        return str(content)
                text = first_choice.get("text")
                if text is not None:
                    return str(text)
            else:
                message = getattr(first_choice, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if content is not None:
                        return str(content)
                text = getattr(first_choice, "text", None)
                if text is not None:
                    return str(text)
    except Exception:
        pass

    text = raw.get("text") if isinstance(raw, dict) else getattr(raw, "text", None)
    if text is not None:
        return str(text)

    return str(raw)
