from __future__ import annotations

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import InvalidRequestError, TaskState, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from .agent import Agent
from .caum_layer import CaumStructuralObserver, stable_hash


TERMINAL_STATES = {
    TaskState.completed,
    TaskState.canceled,
    TaskState.failed,
    TaskState.rejected,
}


class Executor(AgentExecutor):
    def __init__(self):
        self.agents: dict[str, Agent] = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        msg = context.message
        if not msg:
            raise ServerError(error=InvalidRequestError(message="Missing message in request"))

        task = context.current_task
        if task and task.status.state in TERMINAL_STATES:
            raise ServerError(error=InvalidRequestError(message=f"Task {task.id} already processed"))

        if not task:
            task = new_task(msg)
            await event_queue.enqueue_event(task)

        context_id = task.context_id
        updater = TaskUpdater(event_queue, task.id, context_id)
        await updater.start_work()

        observer = CaumStructuralObserver(
            workflow="caum_agentbeats_purple",
            agent_id="caum_agentbeats_purple_v0_1",
            task_family="agentbeats_purple_agent",
        )
        observer.observe("a2a_task_started", phase="receive", tool="a2a_task", state={"context": stable_hash(context_id, "context")})

        try:
            agent = self.agents.get(context_id)
            if not agent:
                agent = Agent(observer=observer)
                self.agents[context_id] = agent

            await agent.run(msg, updater)
            if not updater._terminal_state_reached:
                await updater.complete()
            observer.observe("a2a_task_completed", phase="complete", tool="a2a_task", state={"context": stable_hash(context_id, "context")})
        except Exception as exc:
            observer.observe("a2a_task_failed", phase="error", tool="a2a_task", status="error", state={"error": type(exc).__name__})
            await updater.failed(new_agent_text_message(f"Agent error: {type(exc).__name__}", context_id=context_id, task_id=task.id))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())
