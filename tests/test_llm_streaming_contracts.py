import unittest

from ollama_wrapper.llm import LLMMessage, OpenAIProvider


class TestLLMStreamingContracts(unittest.IsolatedAsyncioTestCase):
    def test_openai_stream_contract_sync(self):
        def stream_fn(**kwargs):
            _ = kwargs
            yield "a"
            yield "b"

        provider = OpenAIProvider(chat_stream_fn=stream_fn)
        chunks = list(provider.chat_stream(model="m", messages=[LLMMessage(role="user", content="hi")]))
        self.assertEqual(chunks, ["a", "b"])

    async def test_openai_stream_contract_async(self):
        async def stream_async_fn(**kwargs):
            _ = kwargs
            for chunk in ["x", "y"]:
                yield chunk

        provider = OpenAIProvider(chat_stream_async_fn=stream_async_fn)
        chunks = []
        async for chunk in provider.chat_stream_async(model="m", messages=[LLMMessage(role="user", content="hi")]):
            chunks.append(chunk)
        self.assertEqual(chunks, ["x", "y"])


if __name__ == "__main__":
    unittest.main()
