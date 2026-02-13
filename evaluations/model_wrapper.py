"""
Azure OpenAI Model Wrapper for DeepEval

This module provides a wrapper class to make Azure OpenAI models compatible with DeepEval's
evaluation metrics.
"""

from deepeval.models import DeepEvalBaseLLM


class AzureOpenAIModel(DeepEvalBaseLLM):
    """Wrapper to make Azure OpenAI compatible with DeepEval."""

    def __init__(self, model):
        """
        Initialize the Azure OpenAI model wrapper.

        Args:
            model: Azure OpenAI chat model instance
        """
        self.model = model

    def load_model(self):
        """Load and return the model instance."""
        return self.model

    def generate(self, prompt: str) -> str:
        """
        Generate response from the model.

        Args:
            prompt: Input prompt string

        Returns:
            str: Generated response content
        """
        chat_model = self.load_model()
        response = chat_model.invoke(prompt)
        return response.content

    async def a_generate(self, prompt: str) -> str:
        """
        Async generate response from the model.

        Args:
            prompt: Input prompt string

        Returns:
            str: Generated response content
        """
        chat_model = self.load_model()
        response = await chat_model.ainvoke(prompt)
        return response.content

    def get_model_name(self) -> str:
        """Return the model name identifier."""
        return "azure-gpt-4o"
