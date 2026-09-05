"""One-line, plain-English explanations for the Settings screens.

Used by the "i" info button on:
  * Settings › BRE Rule Setting › Products      (PRODUCT_DESCRIPTIONS)
  * Settings › …  › rule checklist rows          (describe_rule / RULE_DESCRIPTIONS)

Keep every entry to a single short sentence a non-technical reader understands.
"""
from __future__ import annotations

# ── Loan products ─────────────────────────────────────────────────────────
PRODUCT_DESCRIPTIONS: dict[str, str] = {
    "lap_sbl": "Loan Against Property / Secured Business Loan — credit secured by the "
               "borrower's property and sized on cash-flow plus the property value.",
    "machine": "Term loan to buy plant, machinery or equipment — secured by the asset "
               "and repaid from business cash-flow.",
    "vehicle": "Loan to purchase a commercial or personal vehicle — secured by the "
               "vehicle and repaid in EMIs.",
    "msme": "Working-capital or term loan for a micro, small or medium enterprise, "
            "underwritten mainly on bank and GST cash-flow.",
}

# ── Rule catalogue (data-source rules + product rules) ────────────────────
# Keyed by the exact rule label. A label that appears in more than one
# catalogue only needs one entry here.
RULE_DESCRIPTIONS: dict[str, str] = {
    # Account Aggregator — consent, account & data quality
    "AA Consent Validation Rule": "Confirms the borrower gave a valid, unexpired Account Aggregator consent for this data pull.",
    "AA Account Linkage Validation Rule": "Checks the bank accounts were actually linked and fetched through the AA, not uploaded by hand.",
    "Bank Account Ownership Validation Rule": "Verifies the statement belongs to the applicant (name / PAN match).",
    "Account Status Validation Rule": "Ensures the bank account is active and not closed, frozen or dormant.",
    "Account Type Eligibility Rule": "Checks the account type (savings / current / OD) is one the product accepts.",
    "Statement Period Validation Rule": "Confirms the statement covers the required date range with no gaps.",
    "Minimum Statement History Rule": "Requires enough months of statement history to judge cash-flow (typically 6+).",
    "Data Completeness Rule": "Checks all mandatory fields — dates, amounts, balances, narration — are present.",
    "Data Freshness Rule": "Requires the statement to be recent, ending within the last few weeks.",
    "Transaction Count Validation Rule": "Ensures the statement has enough transactions to be meaningful.",
    "Duplicate Transaction Rule": "Flags identical repeated entries that suggest a tampered or double-counted statement.",
    "Transaction Date Sequence Rule": "Checks transaction dates run in order and stay inside the statement period.",
    "Transaction Amount Validation Rule": "Verifies every transaction amount is a valid, non-zero number.",
    "Invalid Transaction Rule": "Flags malformed or impossible entries (blank amount, bad balance math).",

    # AA — turnover & balances
    "Minimum Monthly Credit Rule": "Requires average monthly credits (inflows) to clear a minimum bar.",
    "Minimum Monthly Debit Rule": "Checks monthly debits are high enough to show a genuinely operating account.",
    "Monthly Credit Volume Rule": "Measures total monthly money coming in.",
    "Monthly Debit Volume Rule": "Measures total monthly money going out.",
    "Credit-Debit Ratio Rule": "Checks inflows exceed outflows over the statement (ratio > 1).",
    "Monthly Net Cash Flow Rule": "Requires a positive surplus after each month's inflows and outflows.",
    "Average Monthly Balance Rule": "Checks the average end-of-day balance stays above a minimum.",
    "Minimum Monthly Balance Rule": "Flags months where the balance fell below the required floor.",
    "Maximum Monthly Balance Rule": "Notes unusually high balances that may need source-of-funds checks.",
    "Negative Balance Days Rule": "Counts days the account was overdrawn.",
    "Average Bank Balance Rule": "Checks the average bank balance stays above a minimum.",
    "Minimum Bank Balance Rule": "Flags dips below the required minimum bank balance.",
    "Negative Balance Rule": "Flags periods where the account balance went negative.",
    "Overdraft Rule": "Detects use of an overdraft / the account running into overdraft.",
    "Expense-to-Income Ratio Rule": "Checks monthly spending is not too large a share of monthly income.",

    # AA — income
    "Income Identification Rule": "Identifies which credits are genuine income vs transfers or refunds.",
    "Salary Credit Identification Rule": "Detects recurring salary credits and the employer pattern.",
    "Salary Credit Regularity Rule": "Checks salary lands on a regular date each month.",
    "Business Income Identification Rule": "Detects recurring business receipts for self-employed applicants.",
    "Business Credit Regularity Rule": "Checks business inflows arrive consistently, not in one-off spikes.",
    "Income Consistency Rule": "Measures how steady monthly income is month to month.",
    "Income Stability Rule": "Scores overall income stability across the statement.",
    "Income Growth Rule": "Rewards a rising income trend.",
    "Income Decline Rule": "Flags a falling income trend.",
    "Income Volatility Rule": "Flags large swings in monthly income.",
    "Bank Income vs Declared Income Rule": "Compares income seen in the bank with the income the applicant declared.",
    "Bank Statement Income Validation Rule": "Confirms the declared income is backed by real bank credits.",
    "Minimum Monthly Income Rule": "Requires average monthly income to clear a minimum.",
    "Minimum Monthly Turnover Rule": "Requires monthly credit turnover to clear a minimum.",
    "Minimum Annual Turnover Rule": "Requires annualised turnover to clear a minimum.",

    # AA — obligations & affordability
    "Existing EMI Identification Rule": "Finds existing loan EMIs being paid from the account.",
    "EMI Obligation Rule": "Totals the applicant's current monthly EMI commitments.",
    "EMI Payment Regularity Rule": "Checks existing EMIs are paid on time without misses.",
    "EMI-to-Income Ratio Rule": "Checks existing EMIs are an affordable share of income.",
    "Existing Debt Burden Rule": "Assesses the overall weight of current debt on the applicant.",
    "Proposed EMI Affordability Rule": "Checks the applicant can afford the new loan's EMI on top of existing ones.",
    "EMI Coverage Ratio Rule": "Checks surplus cash-flow comfortably covers the proposed EMI.",
    "Existing EMI Obligation Rule": "Totals current monthly EMI outgo.",
    "Existing Loan Count Rule": "Counts how many loans the applicant is already servicing.",
    "Existing EMI Burden Rule": "Measures how much of income already goes to EMIs.",
    "EMI-to-Credit Ratio Rule": "Checks EMIs against total credits, not just identified income.",
    "FOIR Rule": "Fixed-Obligation-to-Income Ratio — total fixed outgo (EMIs + proposed EMI) vs income.",
    "DSCR Rule": "Debt-Service-Coverage Ratio — cash available to service debt vs the debt payments.",
    "Proposed EMI Coverage Rule": "Checks monthly surplus covers the proposed EMI with margin.",
    "Income-to-Loan Ratio Rule": "Checks the loan amount is reasonable against annual income.",
    "Loan-to-Income Rule": "Caps the loan as a multiple of income.",
    "Loan-to-Credit-Volume Rule": "Caps the loan against total credit turnover in the statement.",
    "Loan-to-Turnover Rule": "Caps the loan as a share of business turnover.",
    "Loan-to-Business-Turnover Rule": "Caps the loan against business turnover.",
    "Minimum Loan Eligibility Rule": "Checks the applicant qualifies for at least the minimum ticket size.",
    "Maximum Loan Eligibility Rule": "Computes the largest loan the cash-flow can support.",

    # AA — bounces, cash & anomalies
    "Bounce Count Rule": "Counts cheque / auto-debit bounces — a strong repayment-risk signal.",
    "EMI Bounce Rule": "Flags bounced EMI payments specifically.",
    "Cheque Return Rule": "Counts returned (dishonoured) cheques.",
    "Overdraft Frequency Rule": "Counts how often the account goes into overdraft.",
    "Cash Deposit Threshold Rule": "Flags cash deposits above a threshold that need scrutiny.",
    "Cash Withdrawal Threshold Rule": "Flags large cash withdrawals.",
    "Cash Dependency Rule": "Measures how much of the business runs on cash vs banked money.",
    "Cash Deposit Rule": "Checks the level of cash deposits in the account.",
    "Cash Withdrawal Rule": "Checks the level of cash withdrawals.",
    "High-Value Transaction Rule": "Flags unusually large individual transactions.",
    "Unusual Transaction Amount Rule": "Flags amounts that don't fit the account's normal pattern.",
    "Transaction Frequency Anomaly Rule": "Flags sudden bursts or drops in transaction frequency.",
    "Rapid Sequential Transaction Rule": "Flags many transactions in quick succession (possible layering).",
    "Round-Amount Transaction Rule": "Flags a high share of suspiciously round-figure transactions.",
    "Recurring Transaction Rule": "Identifies genuine recurring payments (rent, subscriptions, EMIs).",
    "Self-Transfer Detection Rule": "Detects transfers between the applicant's own accounts, so they aren't counted as income.",
    "Related-Party Transaction Rule": "Flags heavy dealing with connected people or entities.",
    "Counterparty Concentration Rule": "Flags over-reliance on one payer or payee.",
    "Counterparty Diversity Rule": "Rewards a healthy spread of customers and suppliers.",
    "Business Counterparty Diversity Rule": "Checks the business has a diverse customer base.",
    "Supplier Concentration Rule": "Flags dependence on a single supplier.",
    "Unique Sender Count Rule": "Counts distinct parties sending money in.",
    "Unique Receiver Count Rule": "Counts distinct parties receiving money out.",
    "Suspicious Credit Pattern Rule": "Flags inflow patterns that look manufactured.",
    "Suspicious Debit Pattern Rule": "Flags outflow patterns that look manufactured.",
    "Circular Transaction Rule": "Detects money that loops back to its source (round-tripping).",
    "Income-Expense Mismatch Rule": "Flags spending that can't be explained by visible income.",
    "Cash-Flow Stability Rule": "Scores how steady overall cash-flow is.",
    "Monthly Cash-Flow Rule": "Checks each month ends cash-positive.",
    "Bank Credit Volume Rule": "Totals credits over the statement.",
    "Bank Debit Volume Rule": "Totals debits over the statement.",

    # AA — trends & matching
    "Credit Growth 3-Month Rule": "Checks the 3-month trend in inflows.",
    "Credit Growth 6-Month Rule": "Checks the 6-month trend in inflows.",
    "Debit Growth Rule": "Checks whether outflows are rising faster than inflows.",
    "Balance Deterioration Rule": "Flags a steadily falling balance over time.",
    "Negative Cash-Flow Trend Rule": "Flags a worsening month-on-month cash-flow trend.",
    "Revenue Growth Rule": "Rewards growing business revenue.",
    "Business Revenue Stability Rule": "Scores how stable business revenue is.",
    "Business Revenue Consistency Rule": "Checks revenue arrives consistently across periods.",
    "GST-Linked Credit Validation Rule": "Checks bank credits tagged as GST-related look genuine.",
    "GST Payment Pattern Rule": "Checks GST payments leave the account on a regular schedule.",
    "GST Payment (in-statement) Rule": "Confirms GST is actually being paid from this account.",
    "Bank-to-GST Turnover Matching Rule": "Compares turnover seen in the bank with GST-filed turnover.",
    "GST-Bank Turnover Matching Rule": "Cross-checks GST turnover against bank turnover.",
    "GST-to-Bank Turnover Matching Rule": "Cross-checks GST turnover against bank turnover.",
    "Income-Bank Statement Matching Rule": "Checks declared income matches the bank statement.",
    "Income-GST Turnover Matching Rule": "Checks income lines up with GST-declared turnover.",
    "Business Turnover Consistency Rule": "Checks turnover is consistent across the sources we can see.",
    "Financial Stress Indicator Rule": "Combines bounces, overdrafts and falling balances into a stress signal.",

    # AA — risk scores & decision
    "Transaction Anomaly Rule": "Overall anomaly check across all transactions.",
    "Fraud/Suspicious Activity Rule": "Composite fraud signal from the suspicious-pattern checks.",
    "Fraud / Suspicious Activity Rule": "Composite fraud signal from the suspicious-pattern checks.",
    "Account Behaviour Risk Rule": "Scores how risky the account's overall behaviour looks.",
    "AA Risk Score Rule": "The model's overall risk score for the AA data.",
    "Income Consistency Score Rule": "Threshold check on the income-consistency score.",
    "Cash-Flow Stability Score Rule": "Threshold check on the cash-flow-stability score.",
    "Credit Regularity Score Rule": "Threshold check on how regularly credits arrive.",
    "Model Risk Score Rule": "Checks the ML risk score is within the acceptable band.",
    "Model Confidence Rule": "Requires the model to be confident enough in its prediction.",
    "PD Threshold Rule": "Rejects applicants whose probability of default is above the cutoff.",
    "Risk Score Threshold Rule": "Checks the composite risk score clears the policy threshold.",
    "Credit Score Rule": "Checks the credit score meets the minimum for this product.",
    "High-Risk Applicant Rule": "Marks the applicant high-risk when scores breach the danger band.",
    "Medium-Risk Applicant Rule": "Marks the applicant medium-risk.",
    "Low-Risk Applicant Rule": "Marks the applicant low-risk.",
    "Manual Review Rule": "Sends borderline cases to a human underwriter.",
    "Conditional Approval Rule": "Approves subject to conditions being met.",
    "Auto-Approval Rule": "Auto-approves cases that clear every key rule.",
    "Auto-Rejection Rule": "Auto-rejects cases that fail a critical rule.",
    "Policy Override Rule": "Records a deliberate manual override of the policy outcome.",
    "Final AA Underwriting Decision Rule": "The overall approve / review / reject verdict from the AA data.",

    # AA — misc / product-only
    "UPI Payment Behaviour Rule": "Checks UPI activity looks like normal personal / business use.",
    "UPI Business Transaction Rule": "Confirms UPI is being used for genuine business collections.",
    "BBPS Payment Behaviour Rule": "Checks bill payments (BBPS) are regular and on time.",
    "Account Dormancy Rule": "Flags long stretches with no activity.",
    "Business Eligibility Rule": "Checks the business meets basic eligibility (type, age, activity).",
    "Business / Account Vintage Rule": "Requires the business / account to be old enough.",
    "Applicant Eligibility Rule": "Checks the applicant meets age, residency and KYC basics.",
    "Employment/Business Vintage Rule": "Requires enough years in the current job or business.",

    # Bureau (external)
    "DPD History Rule": "Checks the credit bureau for past due-days on existing loans.",
    "Recent Default Rule": "Rejects if the bureau shows a recent default or write-off.",

    # GST — registration & filing
    "GSTIN Validation Rule": "Confirms the GSTIN is well-formed and belongs to the applicant.",
    "GST Registration Status Rule": "Checks the GST registration is Active, not cancelled or suspended.",
    "GST Registration Rule": "Confirms the business holds a valid GST registration.",
    "GST Registration Vintage Rule": "Requires the GST registration to be old enough.",
    "GST Return Period Validation Rule": "Checks the return periods supplied cover the window we need.",
    "GST Return Filing Regularity Rule": "Measures how consistently GST returns are filed on time.",
    "GST Filing Regularity Rule": "Measures on-time GST filing behaviour.",
    "GST Filing Delay Rule": "Flags habitual late filing of GST returns.",
    "Missed Return Rule": "Counts GST return periods that were never filed.",
    "Late Return Rule": "Counts GST returns filed after the due date.",
    "Return Status Validation Rule": "Checks each return's status (Filed / Late / Not filed).",
    "GST Return Consistency Rule": "Checks the set of filed returns is internally consistent.",

    # GST — turnover & matching
    "GSTR-1 vs GSTR-3B Sales Matching Rule": "Checks sales reported in GSTR-1 match the summary in GSTR-3B.",
    "GSTR-1 vs GSTR-3B Taxable Turnover Matching Rule": "Checks taxable turnover agrees between GSTR-1 and GSTR-3B.",
    "GSTR-1 vs GSTR-3B Matching Rule": "Checks GSTR-1 and GSTR-3B are consistent.",
    "GST Turnover Validation Rule": "Sanity-checks the GST turnover figures.",
    "GST Turnover Rule": "Checks GST turnover against product norms.",
    "Minimum GST Turnover Rule": "Requires GST turnover to clear a minimum.",
    "Monthly Turnover Minimum Rule": "Requires monthly GST turnover to clear a minimum.",
    "Quarterly Turnover Minimum Rule": "Requires quarterly GST turnover to clear a minimum.",
    "Month-on-Month Turnover Growth Rule": "Checks the month-on-month turnover trend.",
    "Quarter-on-Quarter Turnover Growth Rule": "Checks the quarter-on-quarter turnover trend.",
    "Year-on-Year Turnover Growth Rule": "Checks the year-on-year turnover trend.",
    "GST Turnover Growth Rule": "Rewards growing GST turnover.",
    "Turnover Decline Rule": "Flags a fall in turnover over recent periods.",
    "Consecutive Declining Quarters Rule": "Flags several quarters of falling turnover in a row.",
    "Consecutive Declining Quarter Rule": "Flags back-to-back quarters of declining turnover.",
    "GST Turnover Volatility Rule": "Flags large swings in GST turnover.",

    # GST — sales mix
    "B2B Sales Percentage Rule": "Checks the share of business-to-business sales.",
    "B2C Sales Percentage Rule": "Checks the share of business-to-consumer sales.",
    "Export Sales Rule": "Notes the share of export sales (zero-rated).",
    "SEZ Sales Rule": "Notes the share of sales to SEZ units.",
    "Reverse Charge Sales Rule": "Notes sales attracting reverse-charge GST.",

    # GST — tax & ITC
    "GST Tax Liability Rule": "Checks net GST payable looks consistent with turnover.",
    "GST Tax Liability Trend Rule": "Checks the trend in GST tax paid over time.",
    "IGST Validation Rule": "Sanity-checks the IGST amounts.",
    "CGST Validation Rule": "Sanity-checks the CGST amounts.",
    "SGST Validation Rule": "Sanity-checks the SGST amounts.",
    "Cess Validation Rule": "Sanity-checks any GST cess amounts.",
    "ITC Availability Rule": "Checks input tax credit available looks reasonable.",
    "ITC Claim Ratio Rule": "Checks how much of available ITC is actually being claimed.",
    "ITC Reversal Rule": "Flags large or frequent ITC reversals.",
    "Net ITC Validation Rule": "Sanity-checks net ITC after reversals.",
    "ITC-to-Turnover Ratio Rule": "Checks ITC is a sensible proportion of turnover.",

    # GST — buyers
    "Unique Buyer Count Rule": "Counts distinct buyers in GSTR-1.",
    "Unique B2B Buyer Count Rule": "Counts distinct B2B buyers.",
    "Top Buyer Concentration Rule": "Flags over-dependence on the single largest buyer.",
    "Top Buyer Sales Percentage Rule": "Measures the largest buyer's share of sales.",
    "Buyer Concentration Level Rule": "Bands buyer concentration as Low / Medium / High.",
    "Buyer Concentration Rule": "Flags concentrated customer risk.",
    "Customer Concentration Risk Rule": "Assesses the risk from a narrow customer base.",

    # GST — loan sizing & data quality
    "GST Turnover-to-Loan Ratio Rule": "Checks the loan against GST turnover.",
    "Loan-to-GST-Turnover Rule": "Caps the loan as a share of GST turnover.",
    "Maximum Loan Eligibility by GST Turnover Rule": "Computes the largest loan GST turnover supports.",
    "Proposed Loan Amount Rule": "Checks the requested amount is within policy limits.",
    "GST Turnover vs Proposed Loan Rule": "Checks the requested loan is reasonable against GST turnover.",
    "GST Data Completeness Rule": "Checks all needed GST fields and periods are present.",
    "GST Data Consistency Rule": "Checks the GST figures agree across returns.",
    "GST Data Anomaly Rule": "Flags GST values that look wrong or manipulated.",

    # GST — risk & decision
    "GST Risk Flag Rule": "The GST model's overall Low / Medium / High risk flag.",
    "GST Underwriting Score Rule": "Checks the GST underwriting score clears the cutoff.",
    "GST-Based Eligibility Rule": "Decides basic eligibility from the GST profile alone.",
    "GST-Based Risk Classification Rule": "Assigns a risk band from the GST profile.",
    "GST-Based Loan Limit Rule": "Sets a loan ceiling from the GST profile.",
    "GST-Based Manual Review Rule": "Routes GST-borderline cases to manual review.",
    "GST-Based Auto-Rejection Rule": "Auto-rejects on a critical GST failure.",
    "GST-Based Auto-Approval Rule": "Auto-approves when the GST profile is clean.",
    "GST Audit/Compliance Risk Rule": "Flags signs of GST audit or compliance exposure.",
    "GST Filing Behaviour Risk Rule": "Scores risk from filing delays, misses and revisions.",
    "Final GST Underwriting Decision Rule": "The overall approve / review / reject verdict from the GST data.",

    # Property (external)
    "Loan-to-Value (LTV) Rule": "Caps the loan as a percentage of the property's value.",
    "Loan-to-Value Rule": "Caps the loan as a percentage of collateral value.",
    "Property Valuation Rule": "Requires a valid third-party property valuation.",
    "Property Type Eligibility Rule": "Checks the property type (residential / commercial / land) is accepted.",
    "Property Ownership Rule": "Confirms clear ownership and title of the property.",
    "Property Encumbrance Rule": "Checks the property is free of existing charges or liens.",
    "Property Location Rule": "Checks the property is in a serviceable, accepted location.",

    # Machine (external)
    "Machine Cost Validation Rule": "Verifies the quoted machine cost against invoices / market price.",
    "Machine Invoice Validation Rule": "Checks the supplier invoice is genuine and complete.",
    "Machine Valuation Rule": "Requires a valuation for the machine being financed.",
    "Machine Type Eligibility Rule": "Checks the machine type is one the product funds.",
    "New/Used Machine Eligibility Rule": "Checks new-vs-used rules for the machine.",
    "Loan-to-Machine-Value Rule": "Caps the loan as a share of the machine's value.",
    "Margin Money Rule": "Requires the borrower to fund a minimum margin from own money.",
    "Down Payment Rule": "Requires a minimum upfront down payment.",
    "Asset Age Rule": "Limits how old a used asset can be.",
    "Dealer/Supplier Validation Rule": "Verifies the dealer or supplier is genuine and approved.",
    "Supplier Concentration/Verification Rule": "Checks and verifies reliance on the supplier.",

    # Vehicle (external)
    "Vehicle Price-to-Income Rule": "Checks the vehicle price is reasonable against income.",
    "Vehicle Valuation Rule": "Requires a valuation for the vehicle.",
    "Vehicle Type Eligibility Rule": "Checks the vehicle type / class is eligible.",
    "New/Used Vehicle Rule": "Applies the new-vs-used vehicle policy.",
    "Vehicle Age Rule": "Limits how old a used vehicle can be.",
    "Dealer Validation Rule": "Verifies the vehicle dealer is authorised.",
    "OEM Validation Rule": "Checks the manufacturer (OEM) is on the approved list.",
    "Insurance Validation Rule": "Requires valid comprehensive insurance on the vehicle.",
    "Registration/RC Validation Rule": "Checks the vehicle registration certificate details.",
    "Vehicle Cost Validation Rule": "Verifies the on-road cost against the quotation.",

    # BBPS Utility Payment History — mined from the bank statement's own
    # utility-bill line items, not a separate feed (see app.bbps.analysis).
    "BBPS Data Presence Rule": "Confirms at least one BBPS/utility bill payment was found in the statement.",
    "Utility Account Count Rule": "Counts how many distinct utility accounts (electricity, water, gas...) are billed in this statement.",
    "Utility Type Diversity Rule": "Flags an applicant paying 2+ different utility types as a stronger residency/stability signal.",
    "Electricity Bill Payment Rule": "Checks for a recurring electricity bill payment.",
    "Water Bill Payment Rule": "Checks for a recurring water bill payment.",
    "Gas Bill Payment Rule": "Checks for a recurring gas bill payment.",
    "Broadband Bill Payment Rule": "Checks for a recurring broadband/internet bill payment.",
    "Mobile / DTH Bill Payment Rule": "Checks for a recurring mobile or DTH bill payment.",
    "Recurring Utility Payment Rule": "Flags whether any utility bill repeats across 2+ calendar months (vs. a one-off payment).",
    "Utility Bill Punctuality Rule": "Scores 0-100 from the on-time payment ratio, penalised for missed months.",
    "Missed Utility Payment Rule": "Counts calendar months a recurring bill should have been paid but wasn't, based on the statement's own span.",
    "On-Time Payment Ratio Rule": "Months a recurring utility was actually paid ÷ months it was expected to be paid.",
    "Average Bill Amount Consistency Rule": "Checks each utility type's bill amount stays roughly consistent month to month.",
    "Statement Span Sufficiency Rule": "Requires enough calendar months in the statement to judge payment regularity at all.",
    "Utility Payment Frequency Rule": "Total BBPS/utility payments found across the statement period.",
    "Final BBPS Underwriting Signal Rule": "Rolls up punctuality, diversity and missed payments into the BBPS utility-stability signal.",

    # UPI Transaction Data Enrichment — one person's UPI transaction log
    # (see app.upi.analysis).
    "UPI Data Presence Rule": "Confirms at least one readable UPI transaction was found in the file.",
    "Transaction Volume Rule": "Checks the applicant has enough UPI transactions to judge behaviour from.",
    "Payment Success Rate Rule": "Share of transactions that actually succeeded, not failed or stayed pending.",
    "Failed Transaction Rule": "Flags an unusually high count of failed UPI payments.",
    "P2P Counterparty Diversity Rule": "Counts distinct people the applicant has received money from via UPI.",
    "P2M Merchant Diversity Rule": "Counts distinct merchants/QR codes the applicant has paid via UPI.",
    "Recurring Payee Rule": "Flags merchants or people paid across 2+ calendar months — a subscription or regular relationship, not a one-off.",
    "High-Risk MCC Exposure Rule": "Share of merchant spend on higher-risk categories (gambling, quasi-cash, pawn shops).",
    "Weekend Spend Concentration Rule": "How much of merchant (P2M) spend falls on a weekend vs. spread through the week.",
    "Average Ticket Size Rule": "The average amount per UPI transaction across the file.",
    "P2P Lending Velocity Rule": "How much money the applicant sends to individuals per month via UPI — an informal-lending signal.",
    "P2P Borrowing Velocity Rule": "How much money the applicant receives from individuals per month via UPI — an informal-borrowing signal.",
    "Daily Transaction Consistency Rule": "Average UPI transactions per day across the file's date span.",
    "Statement Span Sufficiency Rule": "Requires enough calendar months in the file to judge UPI behaviour regularity at all.",
    "Network Stability Rule": "Rolls up payee/payer diversity and recurring relationships into how established the applicant's UPI network is.",
    "Final UPI Underwriting Signal Rule": "Rolls up reliability, risk and network signals into the overall UPI underwriting signal.",
}


def _fallback(label: str) -> str:
    s = (label or "").strip()
    if s.endswith(" Validation Rule"):
        return f"Validates the {s[:-len(' Validation Rule')].strip().lower()}."
    if s.endswith(" Eligibility Rule"):
        return f"Checks {s[:-len(' Eligibility Rule')].strip().lower()} eligibility."
    if s.endswith(" Threshold Rule"):
        return f"Checks the {s[:-len(' Threshold Rule')].strip().lower()} stays within the allowed threshold."
    if s.endswith(" Rule"):
        return f"Checks the {s[:-len(' Rule')].strip().lower()}."
    return s


def describe_rule(label: str) -> str:
    """One-line explanation for a rule label; a readable fallback if uncurated."""
    return RULE_DESCRIPTIONS.get((label or "").strip(), _fallback(label))
