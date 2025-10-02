import os
from crewai.flow.flow import Flow, start, listen, router, or_
from dotenv import load_dotenv
from agentic_rag.tools.rag_module import rag_tool #, execute_rag_search  # importiamo il tool e la funzione diretta
from pydantic import BaseModel, Field
from agentic_rag.crews.check_crew.check_crew import CheckCrew
from agentic_rag.crews.synthesis_crew.synthesis_crew import SynthesisCrew
from agentic_rag.crews.web_search_crew.web_search_crew import WebSearchCrew

import opik

opik.configure(use_local=True)

from opik.integrations.crewai import track_crewai

track_crewai(project_name="provaaaaa")


load_dotenv()

class GuideOutline(BaseModel):
    question: str = ""
    sector: str = ""
    chunk: dict = {}
    search_summary: str = ""


class RagFlow(Flow[GuideOutline]):
    """Flow principale per il RAG tool"""


    @start("failed")
    def get_user_question(self):
        """Prompt for user input and initialize the flow state.

        Returns
        -------
        GuideOutline
            The updated state containing the selected sector and user question.
        """
        self.state.sector = ["Financial Services"]    # Qui definisci il settore su cui lavorare
        print(f"\n=== RAG Tool on: {self.state.sector} ===\n")
        self.state.question = "Which sectors are covered by ETS?"
        #"As a company operating in the financial sector within the EU, what requirements do I need to comply with under the DORA regulation?"
        #input("Insert your question: ")
        
        return self.state
    
    @router(get_user_question)
    def evaluate_relevance(self):
        """Route the flow based on question relevance to the chosen sector.

        Uses a crew to validate the question. If relevant, returns ``"success"``,
        otherwise returns ``"failed"`` to restart from the beginning.

        Returns
        -------
        str
            Either ``"success"`` or ``"failed"`` depending on the evaluation.
        """
        results = []
        for sector in self.state.sector:
            #temp = CheckCrew().crew().kickoff(inputs={"sector":sector, "question":self.state.question})
            results.append(True)  #(temp['FinalResult'])
        if True in results:
            return "success"
        else:
            return "failed"

    
    @listen("success")
    def rag_search(self):
        """Execute the RAG to retrieve relevant information."""
        # Use the direct function instead of the CrewAI tool
        #chunk = execute_rag_search(self.state.question)
        self.state.chunk = rag_tool(self.state.question)
        # state["rag_answer"] = response

        return self.state


    @listen(rag_search)
    def web_search(self):
        """Execute the Web Search crew to gather additional information."""
        web_result = WebSearchCrew().crew().kickoff(inputs={"question": self.state.question})
        # Convert CrewOutput to string
        self.state.search_summary = str(web_result)
        print(f"\n=== Web Search Summary ===\n{self.state.search_summary}\n")
        print(f"\n{type(self.state.search_summary)}\n")
        return self.state
    
    
    @listen(web_search)
    def synthesize_information(self):
        """Execute the Synthesis crew to combine RAG and web information.

        Parameters
        ----------
        state : GuideOutline
            Current flow state with both RAG and web context populated.
        """
        synthesis_crew = SynthesisCrew().crew()
        final_response = synthesis_crew.kickoff(inputs={
            "question": self.state.question,
            "chunk": self.state.chunk,
            "search_summary": self.state.search_summary
        })
        print(f"\n🤖 Final Answer: {final_response["answer"]}")
        print(f"\n🤖 Final Sources: {final_response["sources"]}")


        return {
            "question": self.state.question,
            "chunk": self.state.chunk,
            "search_summary": self.state.search_summary,
            "output": final_response
        }
    
    @listen(synthesize_information)
    def save_response(self, payload):
        """Save response locally and return payload for Opik eval visibility."""
        print("Saving response")
        return payload
    


def kickoff():
    """Run the guide creator flow"""
    RagFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/complete_guide.md to view it.")

if __name__ == "__main__":
    RagFlow().start()