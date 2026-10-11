"""
Generate synthetic training data for document classification.

Creates labeled examples for 4 document categories:
1. Policy Wording
2. Reinsurance Treaty
3. Claims Documentation
4. Broker Agreement

Uses rule-based template generation with randomised parameter values.
Output: storage/training_data/document_classification_training_<timestamp>.json
"""
import json
import logging
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TRAINING_DATA_DIR = Path("storage/training_data")
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────── Templates ──────────────────────────────────────

POLICY_WORDING_TEMPLATES = [
    """
    MOTOR VEHICLE INSURANCE POLICY

    Policy Number: {policy_num}

    SECTION A - COVERAGE
    This policy provides coverage for loss or damage to the insured vehicle arising from:
    (a) Collision or overturning
    (b) Fire, explosion, or lightning
    (c) Theft or attempted theft
    (d) Malicious damage

    SECTION B - EXCLUSIONS
    The Insurer shall not be liable for:
    1. Wear and tear
    2. Mechanical or electrical breakdown
    3. Driving under the influence of alcohol or drugs

    SECTION C - CONDITIONS
    Sum Insured: ${sum_insured}
    Premium: ${premium} per annum
    Excess: ${excess} per claim

    The Insured must notify any claim within 30 days of the incident.
    """,
    """
    PROPERTY ALL RISKS INSURANCE POLICY

    INSURED: {insured_name}
    POLICY PERIOD: {start_date} to {end_date}

    COVERAGE:
    This policy covers all risks of physical loss or damage to the property described below,
    except as provided in the Exclusions section.

    PROPERTY INSURED:
    Building situated at {address}
    Sum Insured: ${sum_insured}

    PREMIUM: ${premium} annually
    DEDUCTIBLE: ${deductible} per occurrence

    CLAIMS PROCEDURE:
    All claims must be reported within 48 hours with supporting documentation.
    """,
    """
    LIFE ASSURANCE POLICY

    Life Assured: {life_assured}
    Policy Number: {policy_num}

    BENEFITS:
    Death Benefit: ${death_benefit}
    Maturity Benefit: ${maturity_benefit}

    PREMIUMS:
    Monthly Premium: ${premium}
    Premium Payment Term: {term} years

    EXCLUSIONS:
    Death by suicide within 12 months of policy commencement
    Death due to engaging in hazardous activities not disclosed

    BENEFICIARY DESIGNATION:
    The Life Assured may nominate beneficiaries who shall receive the death benefit.
    """,
]

REINSURANCE_TREATY_TEMPLATES = [
    """
    REINSURANCE TREATY

    Between: {reinsured} (the Reinsured)
    And: {reinsurer} (the Reinsurer)

    Treaty Type: Quota Share Treaty
    Treaty Period: {start_date} to {end_date}

    ARTICLE 1 - SCOPE
    The Reinsurer agrees to accept {quota_share}% of all policies written by the Reinsured
    under the following classes of business:
    - Motor Insurance
    - Property Insurance
    - Liability Insurance

    ARTICLE 2 - RETENTION AND CESSION
    The Reinsured shall retain {retention_pct}% of each risk and cede the balance to the Reinsurer.

    ARTICLE 3 - PREMIUM
    Reinsurance Premium: {premium_rate}% of gross written premium
    Commission: {commission_rate}%

    ARTICLE 4 - CLAIMS
    The Reinsurer shall pay its proportionate share of all claims within 30 days of settlement by the Reinsured.
    """,
    """
    EXCESS OF LOSS REINSURANCE AGREEMENT

    Reinsured: {reinsured}
    Reinsurer: {reinsurer}

    Treaty Year: {year}

    SECTION 1 - LIMIT AND RETENTION
    Layer: ${layer_limit} xs ${retention}
    The Reinsurer shall indemnify the Reinsured for the amount of each loss
    in excess of ${retention} up to ${layer_limit}.

    SECTION 2 - PREMIUM
    Provisional Premium: ${provisional_premium}
    Minimum Premium: ${minimum_premium}
    Deposit Premium due: {due_date}

    SECTION 3 - CLAIMS COOPERATION
    The Reinsured shall notify the Reinsurer of all claims exceeding ${notification_threshold}
    within 72 hours.

    SECTION 4 - ARBITRATION
    Any disputes shall be resolved by arbitration in accordance with ARIAS rules.
    """,
]

CLAIMS_DOC_TEMPLATES = [
    """
    MOTOR VEHICLE CLAIM FORM

    Claim Number: {claim_num}
    Policy Number: {policy_num}
    Date of Loss: {loss_date}

    CLAIMANT DETAILS:
    Name: {claimant_name}
    Contact: {contact}

    INCIDENT DETAILS:
    Date and Time: {incident_datetime}
    Location: {location}
    Description: {description}

    VEHICLE DAMAGE:
    Make/Model: {vehicle}
    Registration: {registration}
    Estimated Repair Cost: ${repair_cost}

    POLICE REPORT:
    Report Number: {police_report_num}
    Attending Officer: {officer_name}

    DECLARATION:
    I declare that the information provided is true and accurate to the best of my knowledge.

    Signature: _______________
    Date: {declaration_date}
    """,
    """
    CLAIM ASSESSMENT REPORT

    Claim Reference: {claim_ref}
    Insured: {insured_name}

    ASSESSMENT SUMMARY:
    Loss Adjustor: {adjustor_name}
    Inspection Date: {inspection_date}

    FINDINGS:
    Nature of Loss: {nature_of_loss}
    Cause: {cause}

    VALUATION:
    Sum Insured: ${sum_insured}
    Assessed Loss: ${assessed_loss}
    Less Depreciation: ${depreciation}
    Less Excess: ${excess}

    NET CLAIM AMOUNT: ${net_claim}

    RECOMMENDATION:
    {recommendation}

    This claim {approval_status}.

    Signed: {adjustor_signature}
    Date: {signature_date}
    """,
    """
    CLAIM SETTLEMENT ADVICE

    Settlement Number: {settlement_num}
    Claim Number: {claim_num}

    CLAIMANT: {claimant_name}

    SETTLEMENT BREAKDOWN:
    Assessed Claim Value: ${claim_value}
    Less Policy Excess: ${excess}
    Less Salvage Recovery: ${salvage}

    TOTAL PAYABLE: ${total_payable}

    PAYMENT DETAILS:
    Method: {payment_method}
    Bank: {bank_name}
    Account Number: {account_num}

    Payment Date: {payment_date}

    This settlement is in full and final discharge of all liability under Policy {policy_num}.
    """,
]

SANDBOX_APPLICATION_TEMPLATES = [
    """
    REGULATORY SANDBOX APPLICATION

    Applicant: {insurer_name}
    Date: {effective_date}
    Reference: {policy_num}

    1. SANDBOX APPLICATION OVERVIEW
    {insurer_name} hereby submits this sandbox application to participate in the IPEC
    Regulatory Sandbox Programme. We seek a regulatory waiver for the testing period
    of {start_date} to {end_date} to pilot our proposed InsurTech product.

    2. BOUNDARY CONDITIONS
    The testing period shall be limited to {term} months with a defined boundary condition
    of not more than {max_sum_insured} total sum assured per participant.

    3. EXEMPTION SOUGHT
    We request exemption sought under section 14 of the Insurance Act to allow testing
    of our digital distribution channel without a full broker licence.

    4. CONSENT TO PARTICIPATE
    Customers will provide explicit consent to participate in the sandbox trial before
    enrolment. All participants will be informed of the experimental nature of the product.

    5. GRADUATION PHASE
    Upon successful completion, {insurer_name} will transition to full licensing via the
    graduation phase process outlined in the IPEC Sandbox Guidelines.

    6. EXIT PLAN
    In the event of early termination, our exit plan provides for orderly wind-down of
    all active policies with full refund of unearned premiums.

    Signed: {broker_name}
    Date: {signature_date}
    """,
    """
    IPEC REGULATORY SANDBOX — PARTICIPANT APPLICATION FORM

    Organisation: {broker_name}
    Application Reference: {claim_ref}
    Submission Date: {effective_date}

    SECTION A: PRODUCT DESCRIPTION
    We propose to test a blockchain-based claims processing platform during the testing
    period of {start_date} to {end_date}. This sandbox application relates to the
    automated settlement of micro-insurance claims.

    SECTION B: REGULATORY WAIVER REQUEST
    We seek regulatory waiver of the following requirements for the duration of testing:
    - Minimum capital requirement reduced to USD {minimum_premium}
    - Exemption sought for physical office requirement

    SECTION C: BOUNDARY CONDITIONS
    Maximum policyholders: {commission_payment_days},000
    Maximum sum insured per policy: USD {sum_insured}
    Geographic boundary condition: Harare metropolitan area only

    SECTION D: EXIT PLAN
    Our exit plan ensures all customer data is preserved and policies are transferred
    to a licensed carrier if we do not graduate to full authorisation.

    SECTION E: CONSENT TO PARTICIPATE
    All test participants must sign a consent to participate form prior to enrolment.
    The form explains the regulatory sandbox nature of the product.

    Contact: {contact}
    Address: {address}
    """,
]

SANDBOX_QUARTERLY_TEMPLATES = [
    """
    REGULATORY SANDBOX QUARTERLY PROGRESS REPORT

    Participant: {insurer_name}
    Reporting Period: {period_start} — {period_end}
    Sandbox Reference: {policy_num}

    SECTION 1: EXECUTIVE SUMMARY
    This quarterly progress report covers activities undertaken by {insurer_name}
    during the sandbox quarterly reporting period {period_start} to {period_end}.

    SECTION 2: KPI PERFORMANCE
    The following KPI performance metrics were recorded during the period:
    - Total policies issued: {remittance_days}
    - Premiums collected: USD {gwp}
    - Claims settled: {notice_period}
    - Average settlement time: {commission_payment_days} days
    - Customer satisfaction score: 4.2 / 5.0

    SECTION 3: RISK REGISTER
    The risk register identifies the following key risks:
    1. Technology failure — Medium likelihood, high impact
    2. Regulatory non-compliance — Low likelihood, high impact
    3. Customer data breach — Low likelihood, critical impact

    SECTION 4: OPERATIONAL CHALLENGES
    Operational challenges encountered during {period_start}:
    - Integration with legacy systems delayed deployment by {remittance_days} days
    - Customer onboarding volumes below target

    SECTION 5: AUDIT DETAILS
    An internal audit was conducted on {inspection_date}. Audit details confirmed
    compliance with all sandbox boundary conditions.

    SECTION 6: CUSTOMER COMPLAINTS
    Customer complaints received during the period: {notice_period}
    Resolved within 5 business days: {commission_payment_days}
    Escalated: 0

    Submitted by: {broker_name}
    Date: {signature_date}
    """,
    """
    IPEC SANDBOX QUARTERLY REPORT — Q{term}

    Broker/Insurer: {broker_name}
    Report Date: {effective_date}
    Period: {period_start} to {period_end}

    1. PROGRESS AGAINST MILESTONES
    This sandbox quarterly report demonstrates progress against targets committed
    in our original sandbox application.

    2. KPI PERFORMANCE SUMMARY
    Premium written: USD {gwp}
    Claims ratio: {quota_share}%
    Net promoter score: {retention_pct}

    3. FINANCIAL PERFORMANCE
    Gross written premium: USD {gwp}
    Net written premium: USD {nwp}
    Commission paid: USD {total_commission}

    4. RISK REGISTER UPDATE
    No new critical risks added to the risk register. Two existing risks downgraded
    from high to medium following implementation of controls.

    5. OPERATIONAL CHALLENGES
    Operational challenges this quarter were limited to system downtime of {remittance_days} hours.
    All operational challenges have been documented and mitigated.

    6. AUDIT DETAILS
    Quarterly audit conducted by {adjustor_name}. Audit details available on request.
    No material findings.

    7. CUSTOMER COMPLAINTS
    Total customer complaints for the period: {commission_payment_days}
    All resolved within the {notice_period}-day SLA.

    Authorised signatory: {broker_name}
    Date: {declaration_date}
    """,
]

SANDBOX_EXIT_TEMPLATES = [
    """
    REGULATORY SANDBOX EXIT REPORT

    Participant: {insurer_name}
    Exit Report Reference: {claim_ref}
    Date: {effective_date}

    1. TESTING SUMMARY
    This exit report documents the conclusion of our regulatory sandbox participation
    from {start_date} to {end_date}.

    2. TEST OUTCOME
    The test was successful in demonstrating the viability of our digital insurance
    distribution platform. Test outcome confirms product readiness for full market launch.

    3. GRADUATION PHASE TRANSITION
    {insurer_name} is now entering the graduation phase and has submitted a full
    licence application to IPEC. Transition deployment is scheduled for {end_date}.

    4. ORDERLY EXIT PROCESS
    An orderly exit has been executed for all sandbox participants who chose not to
    transition. Post-exit obligations include:
    - Return of unearned premiums: USD {total_payable}
    - Data deletion within {notice_period} days
    - Surrender of sandbox reference number {policy_num}

    5. REGULATORY COMMUNICATION
    The commission will communicate the outcome of the licence application within
    60 working days. Commission will communicate any conditions attached to graduation.

    6. POST-EXIT MONITORING
    {insurer_name} agrees to post-exit monitoring for a period of {term} months
    following the exit report submission.

    Signed: {broker_name}
    Date: {signature_date}
    """,
    """
    IPEC SANDBOX — EXIT AND GRADUATION REPORT

    Applicant: {broker_name}
    Original Sandbox Ref: {policy_num}
    Exit Report Date: {effective_date}

    SECTION A: TEST OUTCOME
    The test was successful. Over the {term}-month testing period we enrolled
    {commission_payment_days} customers with total premiums of USD {gwp}.
    Test outcome confirms the product met all KPI targets.

    SECTION B: ORDERLY EXIT
    An orderly exit from the sandbox has been effected. All {notice_period} active
    policyholders have been notified and given the option to continue on a standard
    policy basis.

    SECTION C: TRANSITION DEPLOYMENT PLAN
    Our transition deployment plan includes:
    - System migration: {start_date} to {end_date}
    - Staff training completion: {inspection_date}
    - Full commercial launch: {end_date}

    SECTION D: GRADUATION PHASE
    We are entering the graduation phase subject to receipt of full authorisation
    from IPEC. The commission will communicate any conditions by {due_date}.

    SECTION E: POST-EXIT OBLIGATIONS
    Post-exit undertakings:
    1. Maintain records for {term} years
    2. Submit quarterly reports for {commission_payment_days} months post-exit
    3. Notify IPEC of any material changes within {remittance_days} days

    Contact: {contact}
    Date: {declaration_date}
    """,
]

BROKER_AGREEMENT_TEMPLATES = [
    """
    INSURANCE BROKERAGE AGREEMENT

    Between: {insurer_name} (the Insurer)
    And: {broker_name} (the Broker)

    Effective Date: {effective_date}

    ARTICLE 1 - APPOINTMENT
    The Insurer appoints the Broker as its authorized insurance broker for the territory of Zimbabwe.

    ARTICLE 2 - BROKER RESPONSIBILITIES
    The Broker shall:
    (a) Solicit and procure insurance business
    (b) Issue cover notes and policy documents
    (c) Collect premiums from policyholders
    (d) Remit premiums to the Insurer within {remittance_days} days

    ARTICLE 3 - COMMISSION
    Commission Rate:
    - Motor Insurance: {motor_commission}%
    - Property Insurance: {property_commission}%
    - Life Insurance: {life_commission}%

    Commission shall be paid monthly within {commission_payment_days} days of premium receipt.

    ARTICLE 4 - AUTHORITY LIMITS
    Maximum Sum Insured per Risk: ${max_sum_insured}
    Binding Authority: ${binding_authority_limit}

    ARTICLE 5 - TERMINATION
    Either party may terminate this agreement with {notice_period} days written notice.
    """,
    """
    BROKER COMMISSION STATEMENT

    Broker: {broker_name}
    Statement Period: {period_start} to {period_end}

    PREMIUM SUMMARY:
    Gross Written Premium: ${gwp}
    Less Returns/Cancellations: ${returns}
    Net Written Premium: ${nwp}

    COMMISSION CALCULATION:
    Motor ({motor_rate}%): ${motor_commission}
    Property ({property_rate}%): ${property_commission}
    Life ({life_rate}%): ${life_commission}

    TOTAL COMMISSION DUE: ${total_commission}

    Less Previous Advances: ${advances}

    NET PAYABLE: ${net_payable}

    Payment will be made via bank transfer within 7 working days.
    """,
]


# ─────────────────────────── Generator ──────────────────────────────────────

def _rand_date(delta_min: int = 0, delta_max: int = 365) -> str:
    return (datetime.now() + timedelta(days=random.randint(delta_min, delta_max))).strftime(
        "%d/%m/%Y"
    )


def _rand_money(lo: int, hi: int) -> str:
    return f"{random.randint(lo, hi):,}"


def generate_rule_based_training_data(samples_per_category: int = 50) -> List[Dict[str, str]]:
    """
    Generate training samples by filling randomised values into document templates.

    Args:
        samples_per_category: Number of samples per category (default 50 → 200 total).

    Returns:
        List of {"text": str, "category": str} dicts, shuffled.
    """
    categories = {
        "policy_wording": POLICY_WORDING_TEMPLATES,
        "reinsurance_treaty": REINSURANCE_TREATY_TEMPLATES,
        "claims_documentation": CLAIMS_DOC_TEMPLATES,
        "broker_agreement": BROKER_AGREEMENT_TEMPLATES,
        "sandbox_application": SANDBOX_APPLICATION_TEMPLATES,
        "sandbox_quarterly_report": SANDBOX_QUARTERLY_TEMPLATES,
        "sandbox_exit_report": SANDBOX_EXIT_TEMPLATES,
    }

    training_data: List[Dict[str, str]] = []

    for category_name, templates in categories.items():
        logger.info("Generating %d samples for: %s", samples_per_category, category_name)
        for _ in range(samples_per_category):
            template = random.choice(templates)
            try:
                filled = template.format(
                    policy_num=f"POL-{random.randint(1000, 9999)}",
                    claim_num=f"CLM-{random.randint(1000, 9999)}",
                    settlement_num=f"SET-{random.randint(1000, 9999)}",
                    claim_ref=f"REF-{random.randint(10000, 99999)}",
                    insured_name=random.choice(
                        ["John Doe", "ABC Limited", "Jane Smith", "XYZ Corporation"]
                    ),
                    life_assured=random.choice(["John Doe", "Jane Smith", "Robert Brown"]),
                    claimant_name=random.choice(["John Doe", "Jane Smith", "Robert Brown"]),
                    reinsured=random.choice(
                        ["First Mutual Zimbabwe", "Old Mutual", "Zimre Holdings"]
                    ),
                    reinsurer=random.choice(["Munich Re", "Swiss Re", "Hannover Re"]),
                    insurer_name=random.choice(
                        ["First Mutual Zimbabwe", "Old Mutual", "Zimre Holdings"]
                    ),
                    broker_name=random.choice(
                        ["ABC Insurance Brokers", "XYZ Risk Solutions", "Prime Brokers Limited"]
                    ),
                    adjustor_name=random.choice(
                        ["Peter Loss Adjustor", "Sarah Assessment Ltd"]
                    ),
                    sum_insured=_rand_money(10_000, 500_000),
                    death_benefit=_rand_money(50_000, 500_000),
                    maturity_benefit=_rand_money(30_000, 300_000),
                    premium=_rand_money(500, 5_000),
                    excess=_rand_money(100, 1_000),
                    deductible=_rand_money(100, 1_000),
                    repair_cost=_rand_money(500, 10_000),
                    claim_value=_rand_money(1_000, 50_000),
                    assessed_loss=_rand_money(1_000, 50_000),
                    depreciation=_rand_money(100, 5_000),
                    salvage=_rand_money(0, 2_000),
                    net_claim=_rand_money(1_000, 40_000),
                    total_payable=_rand_money(1_000, 40_000),
                    quota_share=random.randint(20, 50),
                    retention_pct=random.randint(50, 80),
                    premium_rate=random.randint(5, 15),
                    commission_rate=random.randint(10, 25),
                    motor_commission=random.randint(10, 20),
                    property_commission=random.randint(12, 22),
                    life_commission=random.randint(15, 30),
                    motor_rate=random.randint(10, 20),
                    property_rate=random.randint(12, 22),
                    life_rate=random.randint(15, 30),
                    layer_limit=_rand_money(500_000, 2_000_000),
                    retention=_rand_money(50_000, 200_000),
                    provisional_premium=_rand_money(50_000, 200_000),
                    minimum_premium=_rand_money(25_000, 100_000),
                    notification_threshold=_rand_money(50_000, 200_000),
                    gwp=_rand_money(500_000, 5_000_000),
                    returns=_rand_money(10_000, 100_000),
                    nwp=_rand_money(450_000, 4_900_000),
                    total_commission=_rand_money(50_000, 500_000),
                    advances=_rand_money(0, 100_000),
                    net_payable=_rand_money(40_000, 450_000),
                    max_sum_insured=_rand_money(100_000, 500_000),
                    binding_authority_limit=_rand_money(50_000, 200_000),
                    remittance_days=random.choice([7, 14, 30]),
                    commission_payment_days=random.choice([7, 14, 30]),
                    notice_period=random.choice([30, 60, 90]),
                    term=random.choice([10, 15, 20, 25]),
                    start_date=_rand_date(-365, 0),
                    end_date=_rand_date(1, 365),
                    loss_date=_rand_date(-90, -1),
                    incident_datetime=(
                        datetime.now() - timedelta(days=random.randint(1, 90))
                    ).strftime("%d/%m/%Y %H:%M"),
                    inspection_date=datetime.now().strftime("%d/%m/%Y"),
                    declaration_date=datetime.now().strftime("%d/%m/%Y"),
                    signature_date=datetime.now().strftime("%d/%m/%Y"),
                    payment_date=datetime.now().strftime("%d/%m/%Y"),
                    effective_date=datetime.now().strftime("%d/%m/%Y"),
                    due_date=datetime.now().strftime("%d/%m/%Y"),
                    period_start=datetime.now().strftime("%B %Y"),
                    period_end=(datetime.now() + timedelta(days=30)).strftime("%B %Y"),
                    year=datetime.now().year,
                    address=f"{random.randint(1, 999)} Main Street, Harare",
                    location=random.choice(
                        ["Harare CBD", "Borrowdale", "Eastlea", "Belvedere"]
                    ),
                    description=random.choice(
                        [
                            "Collision with another vehicle",
                            "Fire damage to building",
                            "Theft of vehicle",
                            "Water damage from burst pipe",
                        ]
                    ),
                    vehicle=random.choice(
                        ["Toyota Corolla", "Honda Fit", "Nissan March", "Mazda Demio"]
                    ),
                    registration=f"A{random.randint(100, 999)}-{random.randint(100, 999)}",
                    police_report_num=f"PR{random.randint(10000, 99999)}",
                    officer_name=random.choice(["Sgt. Moyo", "Const. Ncube", "Insp. Dube"]),
                    nature_of_loss=random.choice(
                        ["Accidental damage", "Fire", "Theft", "Natural disaster"]
                    ),
                    cause=random.choice(
                        [
                            "Driver error",
                            "Third party negligence",
                            "Electrical fault",
                            "Weather",
                        ]
                    ),
                    recommendation=random.choice(
                        [
                            "Claim approved for settlement",
                            "Further investigation required",
                            "Partial settlement recommended",
                        ]
                    ),
                    approval_status=random.choice(
                        ["is approved", "requires further review", "is partially approved"]
                    ),
                    adjustor_signature="_____________",
                    contact=f"+263 {random.randint(700, 799)} {random.randint(100, 999)} {random.randint(100, 999)}",
                    payment_method=random.choice(["EFT", "Cheque", "Mobile Money"]),
                    bank_name=random.choice(
                        ["CBZ Bank", "FBC Bank", "Stanbic Bank", "NMB Bank"]
                    ),
                    account_num=f"{random.randint(10000, 99999)}-{random.randint(1000, 9999)}",
                )
                training_data.append({"text": filled.strip(), "category": category_name})
            except KeyError as exc:
                logger.warning("Template key error for %s: %s — skipping sample", category_name, exc)

    random.shuffle(training_data)
    logger.info("Generated %d total training samples", len(training_data))
    return training_data


def save_training_data(training_data: List[Dict[str, str]]) -> str:
    """Serialise training data to a timestamped JSON file."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = TRAINING_DATA_DIR / f"document_classification_training_{timestamp}.json"

    category_counts: Dict[str, int] = {}
    for item in training_data:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1

    payload: Dict[str, Any] = {
        "data": training_data,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_samples": len(training_data),
        "categories": list(category_counts.keys()),
        "samples_per_category": category_counts,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    logger.info("Training data saved to: %s", output_path)
    return str(output_path)


def main() -> None:
    """Entry point: generate and save 200 labelled training samples."""
    print("\n" + "=" * 60)
    print("DOCUMENT CLASSIFICATION TRAINING DATA GENERATION")
    print("=" * 60)

    training_data = generate_rule_based_training_data(samples_per_category=50)
    output_path = save_training_data(training_data)

    print(f"\nGenerated {len(training_data)} training samples")
    print(f"Saved to: {output_path}")
    print("\nNext step: python scripts/train_document_classifier.py")


if __name__ == "__main__":
    main()
