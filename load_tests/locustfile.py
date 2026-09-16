import json
import random
from pathlib import Path

from locust import HttpUser, between, task


COMMENTS_FILE = (
    Path(__file__).parent
    / "test_comments.json"
)


with COMMENTS_FILE.open(
    "r",
    encoding="utf-8",
) as file:
    COMMENTS = json.load(file)


class ModerationUser(HttpUser):
    """
    Simulates a user sending comments
    to the moderation service.
    """

    wait_time = between(
        0.5,
        2.0,
    )

    @task
    def moderate_comment(self):

        comment = random.choice(
            COMMENTS
        )

        payload = {
            "user_id": 1,
            "text": comment,
        }

        with self.client.post(
            "/moderate",
            json=payload,
            name="/moderate",
            catch_response=True,
        ) as response:

            if response.status_code != 200:

                response.failure(
                    f"HTTP {response.status_code}: "
                    f"{response.text}"
                )

                return

            data = response.json()

            if "decision" not in data:

                response.failure(
                    "Missing decision"
                )

                return

            if "prediction_id" not in data:

                response.failure(
                    "Missing prediction_id"
                )

                return

            response.success()