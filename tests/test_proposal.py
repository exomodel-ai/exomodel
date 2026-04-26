import pytest
from exomodel.exomodel import ExoModel

class Proposal(ExoModel):
    @classmethod
    def get_rag_sources(cls):
        return ["tests/rag_proposal.md"]
    client: str = "" # client name
    segment: str = "" # client business segment
    business_challenge: str = "" # what is the problem to be solved
    solution: str = "" # which solution / product is offered based on the vendor portfolio
    solution_objective: str = "" # what's the specific objective of the solution
    solution_description: str = "" # describe the solution, steps and deliverables
    duration: int = 0 # duration in months
    pricing: int = 0 # how much it costs
    payment_conditions: str = "" # how it will be payed
    proposal_validity: int = 0 # how many days the proposal remains valid    

    def create_example(self):
        """Populates the proposal with a high-quality retail strategy example."""
        self.client = "RetailFlow"
        self.segment = "Retail & E-commerce"
        self.business_challenge = (
            "The client faces 'Analysis Paralysis' regarding AI investment. While they have "
            "vast customer data, they lack a prioritized execution plan, leading to missed "
            "opportunities in hyper-personalization and inventory optimization."
        )
        self.solution = "AI Strategy"
        self.solution_objective = (
            "To deliver a 'Business-Ready AI Roadmap' that identifies and ranks the top 5 "
            "high-ROI AI use cases, aligned with current technical capabilities and market trends."
        )
        self.solution_description = (
            "Phase 1: Discovery & Stakeholder Alignment (Weeks 1-3): Deep-dive workshops with "
            "Marketing, Logistics, and IT heads to map current pain points. Deliverable: Current "
            "State AI Maturity Assessment.\n"
            "Phase 2: Technical & Data Audit (Weeks 4-5): Evaluation of existing data quality, "
            "cloud infrastructure, and legacy retail systems to determine 'AI Readiness'. "
            "Deliverable: Data Gap Analysis & Infrastructure Report."
        )
        self.duration = 2
        self.pricing = 60000
        self.payment_conditions = (
            "50% upon contract signature to initiate discovery; "
            "50% upon final delivery of the AI Strategy Masterplan."
        )
        self.proposal_validity = 30
        return self

def test_proposal_initialization():
    proposal = Proposal()
    assert proposal.client == ""
    assert proposal.segment == ""
    assert proposal.business_challenge == ""
    assert proposal.solution == ""
    assert proposal.solution_objective == ""
    assert proposal.solution_description == ""
    assert proposal.duration == 0
    assert proposal.pricing == 0
    assert proposal.payment_conditions == ""
    assert proposal.proposal_validity == 0
    assert proposal.get_rag_sources() == ["tests/rag_proposal.md"]

def test_proposal_creation():
    proposal = Proposal()
    prompt = """
client: RetailFlow

segment: Retail & E-commerce

business_challenge: The client faces "Analysis Paralysis" regarding AI investment. While they have vast customer data, they lack a prioritized execution plan, leading to missed opportunities in hyper-personalization and inventory optimization.

solution_objective: To deliver a "Business-Ready AI Roadmap" that identifies and ranks the top 5 high-ROI AI use cases, aligned with current technical capabilities and market trends.

solution_description: Phase 1: Discovery & Stakeholder Alignment (Weeks 1-3): Deep-dive workshops with Marketing, Logistics, and IT heads to map current pain points. Deliverable: Current State AI Maturity Assessment.
Phase 2: Technical & Data Audit (Weeks 4-5): Evaluation of existing data quality, cloud infrastructure, and legacy retail systems to determine "AI Readiness." Deliverable: Data Gap Analysis & Infrastructure Report.
Phase 3: Opportunity Matrix & Prioritization (Weeks 6-7): Scoring of 10+ potential AI use cases (e.g., Dynamic Pricing, Churn Prediction) based on Business Impact vs. Implementation Effort. Deliverable: Prioritized AI Use-Case Matrix (ROI vs. Feasibility).
Phase 4: Final 12-Month Roadmap & Governance (Week 8): Creation of a sequenced implementation plan including estimated costs, team requirements, and vendor recommendations. Deliverable: Executive AI Strategy Masterplan & Implementation Timeline.

duration: 2

pricing: 60000

payment_conditions: 50% upon contract signature to initiate discovery; 50% upon final delivery of the AI Strategy Masterplan.

proposal_validity: 30       
"""
    proposal.update_object(prompt)
    print(proposal.to_ui())
    assert proposal.client == "RetailFlow"

def test_create_example():
    proposal = Proposal()
    proposal.create_example()
    print(proposal.to_ui())
    assert proposal.client == "RetailFlow"

def test_master_prompt():
    proposal = Proposal()
    proposal.create_example()
    proposal.master_prompt(prompt="update the name of the client to RetailFlow Inc")
    print(proposal)
    assert proposal.client == "RetailFlow Inc"

def test_update_field():
    proposal = Proposal()
    proposal.create_example()
    proposal.update_field(field_name="client", prompt="update the name of the client to RetailFlow Inc")
    print(proposal)
    assert proposal.client == "RetailFlow Inc"

def test_update_object():
    proposal = Proposal()
    proposal.create_example()
    proposal.update_object(prompt="update all necessary fields with the name of the client to RetailFlow Inc")
    print(proposal.to_ui())
    assert proposal.client == "RetailFlow Inc"

def test_run_analysis():
    proposal = Proposal()
    proposal.create_example()
    analysis = proposal.run_analysis()
    print(analysis)
    assert analysis != ""

def test_run_filling_instructions():
    proposal = Proposal()
    instructions = proposal.run_filling_instructions()
    print(instructions)
    assert instructions != ""

def test_run_object_prompt():
    proposal = Proposal()
    proposal.create_example()
    response = proposal.run_object_prompt("What's the name of the client? Return only the name.")
    print(response)
    assert response == "RetailFlow"

if __name__ == "__main__":
    test_run_object_prompt()

    