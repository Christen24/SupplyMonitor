"""WebSocket client adapter for Supply Chain Risk Management."""

from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import SCRMAction, SCRMObservation, SCRMState


class SCRMEnvClient(EnvClient[SCRMAction, SCRMObservation, SCRMState]):
    """Client for communicating with the SCRM environment over WebSocket."""

    def _step_payload(self, action: SCRMAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=SCRMObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> SCRMState:
        return SCRMState(**payload)
