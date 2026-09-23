import sys

import mlflow
from mlflow import MlflowClient


TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "toxicity-moderation-model"

CANDIDATE_ALIAS = "candidate"
PRODUCTION_ALIAS = "production"


def main():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)

    client = MlflowClient(
        tracking_uri=TRACKING_URI,
        registry_uri=TRACKING_URI,
    )

    print("=" * 65)
    print("MODEL PRODUCTION PROMOTION")
    print("=" * 65)

    # --------------------------------------------------------
    # Resolve candidate
    # --------------------------------------------------------

    try:
        candidate = client.get_model_version_by_alias(
            MODEL_NAME,
            CANDIDATE_ALIAS,
        )

    except Exception as exc:
        print(
            f"\nERROR: Could not resolve "
            f"@{CANDIDATE_ALIAS}: {exc}"
        )
        sys.exit(1)

    version = str(candidate.version)

    print(f"\nModel     : {MODEL_NAME}")
    print(f"Candidate : version {version}")

    # --------------------------------------------------------
    # Verify promotion metadata
    # --------------------------------------------------------

    tags = candidate.tags or {}

    gate_status = tags.get(
        "promotion_gate"
    )

    if gate_status != "PASSED":

        print(
            "\nBLOCKED: candidate does not have "
            "promotion_gate=PASSED"
        )

        sys.exit(1)

    print("Gate      : PASSED")

    # --------------------------------------------------------
    # Preserve previous production version for audit output
    # --------------------------------------------------------

    previous_version = None

    try:
        previous = (
            client.get_model_version_by_alias(
                MODEL_NAME,
                PRODUCTION_ALIAS,
            )
        )

        previous_version = str(
            previous.version
        )

    except Exception:
        pass

    # --------------------------------------------------------
    # Promote
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=PRODUCTION_ALIAS,
        version=version,
    )

    client.set_model_version_tag(
        name=MODEL_NAME,
        version=version,
        key="deployment_status",
        value="production",
    )

    # --------------------------------------------------------
    # Verify alias
    # --------------------------------------------------------

    production = (
        client.get_model_version_by_alias(
            MODEL_NAME,
            PRODUCTION_ALIAS,
        )
    )

    if str(production.version) != version:

        print(
            "\nERROR: Production alias verification failed."
        )

        sys.exit(1)

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("PROMOTION SUCCESSFUL")
    print("=" * 65)

    if previous_version is None:

        print("Previous production : none")

    else:

        print(
            f"Previous production : "
            f"version {previous_version}"
        )

    print(
        f"Current production  : "
        f"version {version}"
    )

    print(
        f"Candidate alias     : "
        f"version {version}"
    )

    print(
        "\nProduction URI:"
    )

    print(
        f"models:/{MODEL_NAME}"
        f"@{PRODUCTION_ALIAS}"
    )

    print("=" * 65)


if __name__ == "__main__":
    main()