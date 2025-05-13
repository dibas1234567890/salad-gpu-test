

from pydantic import BaseModel, EmailStr
from typing import Optional


class ToReplaceCard(BaseModel):
    current_card_number: Optional[str]
    reason_for_replacement: Optional[str]


class VisaCardApplicationFor(BaseModel):
    new: bool
    replacement: bool
    re_pin: bool
    green_pin: bool
    e_com_online_payment: bool


class VisaCardType(BaseModel):
    deposit: bool
    usd_debit: bool
    usd_travel: bool
    instant_debit: bool
    usd_e_card: bool


class VisaCard(BaseModel):
    application_for: VisaCardApplicationFor
    type: VisaCardType
    to_replace_card: ToReplaceCard


class SanimaEBankingApplicationFor(BaseModel):
    new: bool
    number_change: bool
    add_account: bool
    service_modification: bool
    reset_password: bool


class RequiredService(BaseModel):
    mobile_banking_sms_only: bool
    mobile_banking_sms_and_gprs_app: bool
    internet_banking: bool


class SanimaEBanking(BaseModel):
    application_for: SanimaEBankingApplicationFor
    existing_mobile_number: Optional[str]
    new_account_number: Optional[str]
    required_service: RequiredService


class IbankingServiceRequired(BaseModel):
    view_only: bool
    fund_transfer: bool


class IBanking(BaseModel):
    service_required: IbankingServiceRequired


class ServiceRequest(BaseModel):
    visa_card: bool
    sanima_e_banking: bool
    i_banking: bool


class DigitalServicesForm(BaseModel):
    branch: str
    date: str
    service_request: ServiceRequest
    applicant_name: str
    account_number: str
    mobile_number: str
    email: str
    visa_card: VisaCard
    sanima_e_banking: SanimaEBanking
    i_banking: IBanking
    debit_account_declaration: bool




# User(name="John", last_name="Doe", id=11)