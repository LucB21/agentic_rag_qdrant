import os
from crewai.flow.flow import Flow, start, listen, router, or_
from dotenv import load_dotenv
from agentic_rag.tools.rag_tool import rag_tool  # importiamo il tool definito in rag_tool.py
from pydantic import BaseModel, Field
from agentic_rag.crews.check_crew.check_crew import CheckCrew
from agentic_rag.crews.rag_crew.rag_crew import RagCrew

import opik

opik.configure(use_local=True)

from opik.integrations.crewai import track_crewai

track_crewai(project_name="crewai-opik-demo")


load_dotenv()

class GuideOutline(BaseModel):
    question: str = ""
    sector: str = ""
    #ethics_result: bool = True


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
        self.state.sector = ["Basket", "Francia"]    # Qui definisci il settore su cui lavorare
        print(f"\n=== RAG Tool on: {self.state.sector} ===\n")
        self.state.question = input("Insert your question: ")
        
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
            temp = CheckCrew().crew().kickoff(inputs={"sector":sector, "question":self.state.question})
            results.append(temp['FinalResult'])
        if True in results:
            return "success"
        else:
            return "failed"

    
    @listen("success")
    def rag_search(self, state):
        """Execute the RAG crew to answer the user question.

        Parameters
        ----------
        state : GuideOutline
            Current flow state. The question is read from ``self.state``.
        """
        response = RagCrew().crew().kickoff(inputs={"question":self.state.question})
        print(f"\n🤖 RAG Answer: {response}")
        # state["rag_answer"] = response

        # # Salva la risposta in un file .md
        # os.makedirs("outputs", exist_ok=True)
        # with open("outputs/answer.md", "w", encoding="utf-8") as f:
        #     f.write(f"# Domanda\n{state['question']}\n\n")
        #     f.write(f"# Risposta\n{state['rag_answer']}\n")

        # print("\n📄 Risposta salvata in outputs/answer.md\n")
        #return self.evaluate

    # def evaluate(self, state):
    #     # Usa l'agente rag_evaluator per valutare con RAGAS
    #     crew = Crew()
    #     result = crew.kickoff(task_id="evaluation", inputs=state)

    #     print("\n📊 Risultati valutazione RAGAS:")
    #     print(result["output"])
    #     return state

    # # Bonus 1: ricerca web
    # def web_search(self, state):
    #     crew = Crew()
    #     result = crew.kickoff(task_id="web_search", inputs=state)
    #     state["combined_answer"] = result["output"]
    #     return state

    # # Bonus 2: validazione delle info web
    # def validate_web(self, state):
    #     crew = Crew()
    #     result = crew.kickoff(task_id="validate_web", inputs=state)
    #     state["validated_info"] = result["output"]
    #     return state

def kickoff():
    """Run the guide creator flow"""
    RagFlow().kickoff()
    print("\n=== Flow Complete ===")
    print("Your comprehensive guide is ready in the output directory.")
    print("Open output/complete_guide.md to view it.")

if __name__ == "__main__":
    RagFlow().start()
