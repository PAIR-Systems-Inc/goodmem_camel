r"""GoodMem + CAMEL ChatAgent example.

Four short scenarios that drive GoodMemToolkit through ChatAgent:

    1. Persistent project context across sessions -- an agent stores facts
       with ``goodmem_remember`` and recalls them with ``goodmem_search``
       after its conversation memory is reset.
    2. A two-agent team knowledge pipeline -- a Scribe writes notes into a
       space, an Analyst answers questions from it.
    3. A structured activity log -- entries are written with a ``category``
       in their metadata, and a release manager's toolkit is scoped
       server-side to ``category == "feat"``. The developer sets the scope;
       the model only supplies the query.
    4. Tool-call inspection -- what the release manager actually called.

Every toolkit is scoped to its space at construction time. The model never
chooses a space and never composes a filter expression.

Set ``GOODMEM_API_KEY`` and ``GOODMEM_BASE_URL``, plus the model provider's
key for CAMEL's default model. ``GOODMEM_VERIFY_SSL=false`` is for a local
server with a self-signed certificate only. ``GOODMEM_EMBEDDER_ID`` pins the
embedder; otherwise the first one the server lists is used.
"""

import os
import time

from camel.agents import ChatAgent
from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

from camel_goodmem import GoodMemToolkit

verify_ssl = os.environ.get("GOODMEM_VERIFY_SSL", "true").lower() != "false"

# One toolkit with the admin surface, used only to set up and tear down.
admin = GoodMemToolkit(verify_ssl=verify_ssl, allow_admin_tools=True, allow_delete=True)
embedder_id = os.environ.get("GOODMEM_EMBEDDER_ID") or admin.list_embedders()[0]["embedderId"]

space_id = admin.create_space("camel-goodmem-example", embedder_id)["spaceId"]
team_space_id = admin.create_space("camel-goodmem-example-team", embedder_id)["spaceId"]
tagged_space_id = admin.create_space("camel-goodmem-example-tagged", embedder_id)["spaceId"]
created = [space_id, team_space_id, tagged_space_id]

model = ModelFactory.create(
    model_platform=ModelPlatformType.DEFAULT,
    model_type=ModelType.DEFAULT,
)


def wait_until_searchable(toolkit: GoodMemToolkit, probe: str, timeout: float = 60.0) -> None:
    """Poll on the write path until a just-written memory is retrievable."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if toolkit.goodmem_search(probe, top_k=3)["totalResults"]:
            return
        time.sleep(2)


try:
    # ---- Scenario 1: Persistent project context across sessions ----
    print("\n=== Scenario 1: Persistent project context across sessions ===")
    project = GoodMemToolkit(space_ids=[space_id], verify_ssl=verify_ssl)
    agent = ChatAgent(
        system_message=(
            "You are an engineering team assistant whose long-term memory is "
            "stored in GoodMem. When the user shares a project fact, call "
            "goodmem_remember. When the user asks a question, call "
            "goodmem_search first. Do not answer from your conversational memory."
        ),
        model=model,
        tools=project.get_tools(),
    )
    for turn in [
        "I'm building a customer support assistant for our SaaS product.",
        "The team uses Python 3.12 with FastAPI and Postgres.",
        "For tests we use pytest with at least 80% coverage required.",
    ]:
        print(f"\nUser:  {turn}")
        print(f"Agent: {agent.step(turn).msgs[0].content}")
    wait_until_searchable(project, "coverage requirement")
    agent.reset()
    question = "Remind me what our coverage requirement is."
    print(f"\nUser:  {question}")
    print(f"Agent: {agent.step(question).msgs[0].content}")

    # ---- Scenario 2: Two-agent team knowledge pipeline ----
    print("\n=== Scenario 2: Two-agent team knowledge pipeline ===")
    team = GoodMemToolkit(space_ids=[team_space_id], verify_ssl=verify_ssl)
    scribe = ChatAgent(
        system_message="You are a team Scribe. Store each note verbatim with goodmem_remember and confirm briefly.",
        model=model,
        tools=team.get_tools(),
    )
    for note in [
        "Q2 goal: reduce customer support response time to under 2 hours.",
        "Our main services are auth-service, billing-service, and notifications-service.",
        "Known issue: notifications-service drops messages during high load.",
        "Team retro: the CI pipeline is too slow; we should parallelize tests.",
    ]:
        scribe.step(note)
    wait_until_searchable(team, "CI pipeline")
    analyst = ChatAgent(
        system_message="You are a team Analyst. Answer team questions by calling goodmem_search.",
        model=model,
        tools=GoodMemToolkit(space_ids=[team_space_id], verify_ssl=verify_ssl, allow_write=False).get_tools(),
    )
    question = "What do we know about our services and current priorities?"
    print(f"\nUser:  {question}")
    print(f"Agent: {analyst.step(question).msgs[0].content}")

    # ---- Scenario 3: Structured team activity log ----
    print("\n=== Scenario 3: Structured team activity log ===")
    tagged = GoodMemToolkit(space_ids=[tagged_space_id], verify_ssl=verify_ssl)
    for content, category in [
        ("Added user profile editing to the dashboard.", "feat"),
        ("Built the CSV export feature.", "feat"),
        ("Resolved slow login on the mobile app.", "fix"),
        ("Fixed crash when opening large attachments.", "fix"),
        ("Upgraded Python version across services.", "chore"),
        ("Updated the API reference for billing endpoints.", "docs"),
    ]:
        # Written directly, so the metadata is deterministic.
        tagged.goodmem_remember(content, metadata={"category": category})
    wait_until_searchable(tagged, "CSV export")

    # The release manager's search is scoped to feat entries server-side.
    features_only = GoodMemToolkit(
        space_ids=[tagged_space_id],
        verify_ssl=verify_ssl,
        allow_write=False,
        metadata_filter={"category": "feat"},
    )
    release_manager = ChatAgent(
        system_message=(
            "You are a release manager. Call goodmem_search to find what the team "
            "shipped and report each result's text."
        ),
        model=model,
        tools=features_only.get_tools(),
    )
    question = "Show me the new features we've shipped."
    print(f"\nUser:  {question}")
    response = release_manager.step(question)
    print(f"Agent: {response.msgs[0].content}")

    # ---- Scenario 4: Tool-call inspection ----
    print("\n=== Scenario 4: Tool-call inspection ===")
    for tool_call in response.info.get("tool_calls", []):
        print(f"  {tool_call.tool_name}({tool_call.args})")
finally:
    print("\n=== Cleanup ===")
    for sid in created:
        admin.delete_space(sid)
    remaining = {s["spaceId"] for s in admin.list_spaces()} & set(created)
    print("  all example spaces deleted" if not remaining else f"  WARNING: not deleted: {remaining}")
    admin.close()
