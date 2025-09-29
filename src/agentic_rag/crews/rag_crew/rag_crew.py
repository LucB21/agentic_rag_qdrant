# src/rag_crew/rag_crew.py
from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai import Task
from typing import List

@CrewBase
class RagCrew():
    """Crew per RAG tool"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def rag_researcher(self):
        """Create the RAG researcher agent with access to the retrieval tool.

        Returns
        -------
        crewai.Agent
            Agent responsible for performing RAG searches and crafting answers.
        """
        return Agent(
            config=self.agents_config['rag_researcher'],
            verbose=True
        )

    # @agent 
    # def rag_evaluator(self):
    #     """Create the RAG evaluator agent.

    #     Returns
    #     -------
    #     crewai.Agent
    #         Agent responsible for assessing outputs with evaluation criteria.
    #     """
    #     return Agent(
    #         config=self.agents_config['rag_evaluator'],
    #         verbose=True
    #     )

    @task
    def rag_search(self):
        """Create the RAG search task executed by the researcher agent.

        Returns
        -------
        crewai.Task
            Task that generates an answer with source references.
        """
        return Task(config=self.tasks_config['rag_search'])

    @crew
    def crew(self) -> Crew:
        """Assemble the crew with agents and tasks in sequential process.

        Returns
        -------
        crewai.Crew
            Configured crew to execute RAG search (and optional evaluation).
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
