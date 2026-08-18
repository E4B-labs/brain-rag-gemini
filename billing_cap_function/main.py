"""Budget notification handler for the project billing safety cap."""

import base64
import json
import os

import functions_framework
from cloudevents.http import CloudEvent
from google.api_core import exceptions
from google.cloud import billing_v1


billing_client = billing_v1.CloudBillingClient()


def _project_id() -> str:
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT is required")
    return project_id


@functions_framework.cloud_event
def stop_billing(cloud_event: CloudEvent) -> None:
    """Disable project billing at or above the budget, unless this is a dry run."""
    message = cloud_event.data["message"]
    raw_payload = base64.b64decode(message["data"]).decode("utf-8")
    try:
        payload = json.loads(raw_payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        print("Ignoring malformed budget notification payload.")
        return
    cost_amount = float(payload.get("costAmount", 0))
    budget_amount = float(payload.get("budgetAmount", 0))
    dry_run = bool(payload.get("dryRun", False))
    project_name = f"projects/{_project_id()}"

    print(
        f"Budget notification: project={project_name} "
        f"cost={cost_amount} budget={budget_amount} dry_run={dry_run}"
    )
    if cost_amount < budget_amount:
        print("No action required: current cost is below budget.")
        return
    if dry_run:
        print("Dry run: billing would be disabled; no update was sent.")
        return
    if not _is_billing_enabled(project_name):
        print("Billing is already disabled.")
        return

    try:
        billing_client.update_project_billing_info(
            name=project_name,
            project_billing_info=billing_v1.ProjectBillingInfo(
                billing_account_name=""
            ),
        )
        print(f"Billing disabled for {project_name}.")
    except exceptions.PermissionDenied:
        print("Billing disable failed: service account lacks required IAM roles.")
        raise


def _is_billing_enabled(project_name: str) -> bool:
    try:
        response = billing_client.get_project_billing_info(name=project_name)
        return bool(response.billing_enabled)
    except Exception as error:
        print(f"Billing status lookup failed ({error}); assuming enabled.")
        return True
