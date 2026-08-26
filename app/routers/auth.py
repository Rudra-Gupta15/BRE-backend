import random
import string

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

# Captchas are short-lived and keyed by their own code (demo-grade; a real
# system would key by session/IP and expire them).
_active_captchas: set[str] = set()


def _generate_captcha_code() -> str:
    code = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(5))
    _active_captchas.add(code)
    return code


@router.get("/captcha")
async def get_captcha():
    return {"captchaCode": _generate_captcha_code()}


class LoginBody(BaseModel):
    email: str | None = None
    password: str | None = None
    captchaInput: str | None = None
    captchaCode: str | None = None


@router.post("/login")
async def login(body: LoginBody):
    if not body.email or not body.password:
        raise HTTPException(400, "Please enter both Email ID / Username and Password.")
    if not body.captchaInput or not body.captchaCode or body.captchaInput.lower() != body.captchaCode.lower():
        raise HTTPException(401, "Invalid Captcha code. Please try again.")
    _active_captchas.discard(body.captchaCode)
    return {"user": {"email": body.email, "role": "Admin"}}


class QuickLoginBody(BaseModel):
    role: str = "Admin"


@router.post("/quick-login")
async def quick_login(body: QuickLoginBody):
    mock_email = "admin@satinfinserv.com" if body.role == "Admin" else "user@satinfinserv.com"
    return {"user": {"email": mock_email, "role": body.role}}
