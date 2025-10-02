from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai import Task
from typing import List
from pydantic import BaseModel

class Result(BaseModel):
    answer: str
    sources: List[dict]


@CrewBase
class SynthesisCrew():
    """Crew dedicata alla sintesi delle informazioni"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def information_synthesizer(self):
        """Create the information synthesizer agent.

        Returns
        -------
        crewai.Agent
            Agent responsible for combining RAG and web information.
        """
        return Agent(
            config=self.agents_config['information_synthesizer'],
            verbose=True
        )

    @task
    def synthesize_information(self):
        """Create the information synthesis task.

        Returns
        -------
        crewai.Task
            Task that combines RAG and web information into final answer.
        """
        return Task(
            config=self.tasks_config['synthesize_information'],
            output_json=Result
            )

    @crew
    def crew(self) -> Crew:
        """Assemble the synthesis crew.

        Returns
        -------
        crewai.Crew
            Configured crew to execute information synthesis.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )