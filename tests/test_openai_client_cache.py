import unittest
from unittest import mock


class TestOpenAIClientCache(unittest.TestCase):
    def test_openai_client_is_cached_per_thread_and_endpoint(self):
        # Avoid importing the full completion stack (pulls optional deps).
        # Instead, mock an `openai` module and test the small helper directly.
        from utils import openai_client

        fake_openai = mock.Mock()
        fake_openai.OpenAI.side_effect = [object(), object(), object()]

        with mock.patch.dict("sys.modules", {"openai": fake_openai}):
            c1 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 123}
            )
            c2 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 123}
            )
            self.assertIs(c1, c2)
            self.assertEqual(fake_openai.OpenAI.call_count, 1)

            # Different timeout should create a new client
            c3 = openai_client.get_openai_client(
                {"api_base": "http://localhost:8000/v1", "api_key": "k", "timeout": 456}
            )
            self.assertIsNot(c1, c3)
            self.assertEqual(fake_openai.OpenAI.call_count, 2)


if __name__ == "__main__":
    unittest.main()

