import json
import unittest

from runtime_attestation_redaction import redact_runtime_attestation


class RuntimeAttestationRedactionTests(unittest.TestCase):
    def test_redacts_secret_fields_recursively(self):
        payload = {
            "provider": "vast",
            "instance_id": "instance-1",
            "api_token": "token-value",
            "nested": {
                "private_key": "key-value",
                "reward_address": "alice-address",
            },
            "headers": [
                {"Authorization": "Bearer abc"},
                {"safe": "value"},
            ],
        }

        redacted = redact_runtime_attestation(payload)
        encoded = json.dumps(redacted, sort_keys=True)

        self.assertEqual(redacted["api_token"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["private_key"], "[REDACTED]")
        self.assertEqual(redacted["headers"][0]["Authorization"], "[REDACTED]")
        self.assertEqual(redacted["nested"]["reward_address"], "alice-address")
        self.assertNotIn("token-value", encoded)
        self.assertNotIn("key-value", encoded)
        self.assertNotIn("Bearer abc", encoded)


if __name__ == "__main__":
    unittest.main()
