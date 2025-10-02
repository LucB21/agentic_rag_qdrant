from crewai import Agent, Crew, Process
from crewai.project import CrewBase, agent, task, crew
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
from crewai import Task
from typing import List

@CrewBase
class WebSearchCrew():
    """Crew dedicata alla ricerca web"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def web_researcher(self):
        """Create the web researcher agent.

        Returns
        -------
        crewai.Agent
            Agent responsible for web searches and gathering external information.
        """
        return Agent(
            config=self.agents_config['web_researcher'],
            verbose=True,
            tools=[SerperDevTool(verify = False, n_results=3)]
            )

    @task
    def web_search(self):
        """Create the web search task.

        Returns
        -------
        crewai.Task
            Task that performs web search and gathers external information.
        """
        return Task(config=self.tasks_config['web_search'])

    @crew
    def crew(self) -> Crew:
        """Assemble the web search crew.

        Returns
        -------
        crewai.Crew
            Configured crew to execute web search.
        """
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )