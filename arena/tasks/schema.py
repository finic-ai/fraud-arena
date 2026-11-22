"""Fixed vocabularies shared by the tasks: typology families, customer roles, and the map from the world's fraud
scenarios to them."""

TYPOLOGIES = ["ato", "scam_victim", "mule", "synthetic_identity", "first_party", "card_compromise", "promo_abuse",
              "check_fraud", "other", "none"]
ROLES = ["perpetrator", "victim", "mule", "legitimate"]
DISPOSITIONS = ["fraud", "not_fraud"]
ALERT_ACTIONS = ["close_no_action", "monitor", "contact_customer", "step_up_auth", "restrict_account", "close_account",
                 "reimburse_victim", "file_sar"]
INCIDENT_ACTIONS = ALERT_ACTIONS + ["block_device", "block_ip", "block_payee", "block_merchant", "block_originator",
                                    "reissue_cards", "write_rule"]

# scenario -> (typology family, kind). kind says how a customer touched by the scenario relates to it:
#   victim: the fraudster acts on the customer's account or card, or the customer is scammed into paying
#   self: the customer's own fraud (first-party)
#   mule: the customer's account moves other people's stolen money
#   opened: the fraudster opened the account in a stolen or synthetic identity
#   cover: a ring's cover activity on its own accounts (not a family of its own)
SCENARIOS = {
    # account takeover
    "ato_credential_stuffing": ("ato", "victim"), "ato_phishing": ("ato", "victim"), "sim_swap_ato": ("ato", "victim"),
    "exchange_ato_drain": ("ato", "victim"), "loyalty_ato": ("ato", "victim"), "phishing_campaign": ("ato", "victim"),
    "merchant_payout_ato": ("ato", "victim"), "courier_ato": ("ato", "victim"),
    "delivery_customer_ato": ("ato", "victim"), "r2_phishing_campaign": ("ato", "victim"),
    "r2_courier_ato": ("ato", "victim"), "r2_delivery_customer_ato": ("ato", "victim"),
    "r2_ato_phishing": ("ato", "victim"), "r3_corporate_ato": ("ato", "victim"), "acats_fraud_out": ("ato", "victim"),
    # R7 pulls money from breached bank logins into Dovrex accounts; at Brindlemark its customers are the victims
    "r7_ach_return": ("ato", "victim"),
    # card compromise
    "vishing_card_details": ("card_compromise", "victim"), "r2_vishing_card_details": ("card_compromise", "victim"),
    "card_testing": ("card_compromise", "victim"), "r6_card_testing": ("card_compromise", "victim"),
    "cnp_fraud": ("card_compromise", "victim"), "r6_cnp_fraud": ("card_compromise", "victim"),
    "r10_fake_store_orders": ("card_compromise", "victim"),
    # scams (the customer authorises the payment)
    "romance": ("scam_victim", "victim"), "pig_butchering": ("scam_victim", "victim"),
    "investment_scam": ("scam_victim", "victim"), "purchase_scam": ("scam_victim", "victim"),
    "impersonation_safe_account": ("scam_victim", "victim"), "tech_support_refund": ("scam_victim", "victim"),
    "grandparent": ("scam_victim", "victim"), "elder_theft_caregiver": ("scam_victim", "victim"),
    "r4_pig_butchering": ("scam_victim", "victim"), "r4_romance": ("scam_victim", "victim"),
    "r4_platform": ("scam_victim", "victim"), "r9_grandparent": ("scam_victim", "victim"),
    "r9_tech_support": ("scam_victim", "victim"), "bec_vendor": ("scam_victim", "victim"),
    "r3_bec_vendor": ("scam_victim", "victim"), "invoice_fraud": ("scam_victim", "victim"),
    "payroll_diversion": ("scam_victim", "victim"), "r3_payroll_diversion": ("scam_victim", "victim"),
    "triangulation": ("card_compromise", "victim"), "r6_triangulation": ("card_compromise", "victim"),
    # mules
    "money_mule": ("mule", "mule"), "job_scam_mule": ("mule", "mule"), "ring_mule_flow": ("mule", "mule"),
    "r8_job_ads": ("mule", "mule"), "r3_shell_account": ("mule", "mule"),
    # first-party
    "friendly_fraud": ("first_party", "self"), "deposit_fraud": ("first_party", "self"),
    "loan_stacking": ("first_party", "self"), "refund_abuse": ("first_party", "self"),
    "first_party_ach_return": ("first_party", "self"), "courier_fraud": ("first_party", "self"),
    "r10_refund_service": ("first_party", "self"), "r10_courier_rental": ("first_party", "self"),
    "structuring": ("other", "self"),
    # accounts opened by fraudsters
    "r1_bank_hopping": ("promo_abuse", "opened"), "r10_promo_farm": ("promo_abuse", "self"),
    "r5_synthetic_bustout": ("synthetic_identity", "opened"), "r5_collusive_merchant": ("synthetic_identity", "opened"),
    "r5_check_kiting": ("check_fraud", "opened"), "new_account_id_theft": ("synthetic_identity", "opened"),
    "ach_return_fraud": ("synthetic_identity", "opened"), "acats_fraud": ("synthetic_identity", "opened"),
    "r10_fake_restaurant": ("synthetic_identity", "opened"), "r6_setup": ("other", "opened"),
    "ring_cover_activity": (None, "cover"), "ring_operation": (None, "cover"),
}

# The money the customer itself sends out in a scam (a victim's own payment); receiving alone is not victimhood.
PAYING_EVENTS = {"p2p_send", "personal_wire", "ach_transfer_out", "ach_debit", "cash_withdrawal", "payee_add",
                 "card_purchase", "transfer_internal", "login", "bank_call", "wire_out"}
