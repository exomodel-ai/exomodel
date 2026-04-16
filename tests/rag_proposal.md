# PROPOSAL

## Definition

**Entity:** A **Proposal** is a formal document and data structure used to pitch AI consulting services. It maps a client’s industry-specific challenges to a structured solution (Strategy, Execution, or Labs). The assistant must **interpret and classify** the user's intent into the official portfolio, ensuring no fields are left at zero.

## Portfolio & Semantic Mapping (Anti-Literal Logic)

| Official Solution | Synonyms & Keywords (User Input) | Default Duration | Default Pricing (USD) | Payment Terms |
| --- | --- | --- | --- | --- |
| **AI Strategy** | Roadmap, Assessment, Discovery, Readiness, Prioritization. | 2 months | 60,000 | 50% Upfront / 50% Final |
| **AI Program Execution** | Governance, PMO, Squads, Partner, Management, Scaling. | 12 months | 20,000 (Monthly) | Monthly (Post-paid) |
| **AI Labs** | **Laboratory**, Pilots, MVP, Experimentation, Validation, Prototype. | 6 months | 250,000 | 4 Installments (Standard) |

## Objective

The objective is to formalize an AI engagement by aligning the vendor’s expertise with the client’s segment-specific needs. It serves as the source of truth for scope, methodology, and commercial values.

## Relationships

* **Parent Entity:** Client / Account.
* **Relationship Type:** One-to-Many.

## Fields

* **client:** Legal or commercial name of the organization.
* **segment:** Industry or vertical market (e.g., Agribusiness, Retail).
* **business_challenge:** Description of the pain point or AI gap to be solved.
* **solution:** **[MANDATORY]** Strictly use: **AI Strategy**, **AI Program Execution**, or **AI Labs**.
* **solution_objective:** The concrete "North Star" goal (use action verbs).
* **solution_description:** Breakdown of phases, methodology, and key deliverables.
* **duration:** Total project cycle in months (Integer).
* **pricing:** Total project cost or monthly fee in USD (Integer).
* **payment_conditions:** Specific schedule for invoicing.
* **proposal_validity:** Days the terms remain valid (Default: 30).

---

## Instructions for Inference

### Field: solution (Mapping Rule)

DO NOT copy the user's terminology literally.

* If the user mentions "Laboratory" or "Experimental pilots", classify as **AI Labs**.
* If the user mentions "Roadmap" or "Prioritization", classify as **AI Strategy**.
* If the user mentions "Running the program" or "Governance", classify as **AI Program Execution**.

### Field: pricing & duration (Default Values)

If the user prompt **does not specify** values, you MUST infer them from the Portfolio Mapping table based on the classified solution. **Never return 0.**

* Example: For a "Laboratory" request, set `pricing: 250000` and `duration: 6`.

### Field: payment_conditions (Standardization)

Automatically populate based on the classified `solution`:

* **AI Strategy:** 50% upon signature, 50% upon delivery.
* **AI Program Execution:** Monthly fee paid at the end of each month executed.
* **AI Labs:** 4 equal installments: Signature, Blueprint, Validation, and Final Delivery.

---

## Recommendations (Do's and Don'ts)

### Do's:

* **Interpret Intent:** Convert "laboratory of AI" or "AI pilot lab" into the official product name **AI Labs**.
* **Segment Intelligence:** Use the client's segment to customize the `solution_description` (e.g., for Agribusiness, mention "plantation data" or "yield optimization").
* **Fill Gaps:** If the prompt is missing details, use the RAG's default values to ensure a complete data structure.

### Don'ts:

* **Zero Values:** Never leave `pricing`, `duration`, or `proposal_validity` as 0 or empty.
* **Literal Copying:** Do not create new product names like "AI Laboratory"; stay within the three official portfolio items.

---

## EXAMPLES (Contextual Inference)

### EXAMPLE 1: AI Labs (Inferred from "Laboratory")

### EXAMPLE 1: AI Labs (Inferred from "Laboratory")

EXAMPLE 1: AI Labs (Agribusiness Focus)
client: AgroBusiness

segment: Agribusiness (Corn cultivation & Processing)

business_challenge: The client has fragmented AI experiments in soil monitoring and office logistics but lacks a unified, validated framework. This results in inconsistent crop yield predictions and data silos between the field and the processing plant.

solution: AI Labs

solution_objective: To engineer and validate a high-fidelity "Corn Yield Predictive Model" by fusing field IoT data with historical processing metrics, achieving a minimum 85% prediction accuracy.

solution_description: Phase 1: Technical & Data Blueprint (Month 1): Audit of current sensor infrastructure (soil/weather) and office ERP data. Deliverable: AI Infrastructure Roadmap & Success Metric Definition.
Phase 2: Data Engineering & Sensor Fusion (Months 2-3): Cleaning and normalizing telemetry data from corn fields. We synchronize temporal field data with processing plant throughput. Deliverable: Unified Agricultural Data Lake & Data Quality Audit.
Phase 3: Iterative MVP Development (Months 4-5): Training and testing Machine Learning models for corn yield optimization. We run "Champion-Challenger" algorithm tests. Deliverable: Functional AI Prototype (MVP) & Model Performance Dashboard.
Phase 4: Field Validation & Results Report (Month 6): Stress-testing the model against real-time harvest data to certify consistency. Deliverable: Final Validation Certificate & Full-Scale Deployment Blueprint.

duration: 6

pricing: 250000

payment_conditions: 4 equal installments: 25% at Signature, 25% upon Blueprint approval, 25% upon MVP Validation, and 25% upon Final Delivery.

proposal_validity: 30

### EXAMPLE 2: AI Strategy (Inferred from "Roadmap")

client: RetailFlow

segment: Retail & E-commerce

business_challenge: The client faces "Analysis Paralysis" regarding AI investment. While they have vast customer data, they lack a prioritized execution plan, leading to missed opportunities in hyper-personalization and inventory optimization.

solution: AI Strategy

solution_objective: To deliver a "Business-Ready AI Roadmap" that identifies and ranks the top 5 high-ROI AI use cases, aligned with current technical capabilities and market trends.

solution_description: Phase 1: Discovery & Stakeholder Alignment (Weeks 1-3): Deep-dive workshops with Marketing, Logistics, and IT heads to map current pain points. Deliverable: Current State AI Maturity Assessment.
Phase 2: Technical & Data Audit (Weeks 4-5): Evaluation of existing data quality, cloud infrastructure, and legacy retail systems to determine "AI Readiness." Deliverable: Data Gap Analysis & Infrastructure Report.
Phase 3: Opportunity Matrix & Prioritization (Weeks 6-7): Scoring of 10+ potential AI use cases (e.g., Dynamic Pricing, Churn Prediction) based on Business Impact vs. Implementation Effort. Deliverable: Prioritized AI Use-Case Matrix (ROI vs. Feasibility).
Phase 4: Final 12-Month Roadmap & Governance (Week 8): Creation of a sequenced implementation plan including estimated costs, team requirements, and vendor recommendations. Deliverable: Executive AI Strategy Masterplan & Implementation Timeline.

duration: 2

pricing: 60000

payment_conditions: 50% upon contract signature to initiate discovery; 50% upon final delivery of the AI Strategy Masterplan.

proposal_validity: 30
