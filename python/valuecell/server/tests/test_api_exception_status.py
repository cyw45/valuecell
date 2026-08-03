from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from valuecell.server.api.exceptions import (
    APIException,
    api_exception_handler,
    general_exception_handler,
    http_exception_handler,
    validation_exception_handler,
)


def test_http_exception_preserves_forbidden_status_and_message():
    app = FastAPI()
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    @app.get("/forbidden")
    def forbidden():
        raise HTTPException(status_code=403, detail="工作区尚未开通或服务已到期")

    response = TestClient(app, raise_server_exceptions=False).get("/forbidden")

    assert response.status_code == 403
    assert response.json()["msg"] == "工作区尚未开通或服务已到期"
    assert response.json()["data"] is None


def test_http_exception_preserves_structured_validation_detail():
    app = FastAPI()
    app.add_exception_handler(HTTPException, http_exception_handler)

    @app.get("/invalid-demo-account")
    def invalid_demo_account():
        raise HTTPException(
            status_code=422,
            detail={
                "code": "okx_demo_connection_invalid",
                "error_code": "credential_or_permission_error",
            },
        )

    response = TestClient(app, raise_server_exceptions=False).get(
        "/invalid-demo-account"
    )

    assert response.status_code == 422
    assert response.json()["code"] == 422
    assert response.json()["detail"] == {
        "code": "okx_demo_connection_invalid",
        "error_code": "credential_or_permission_error",
    }