import pytest

from src.config.settings import settings
from src.services.memory_service import HttpMemoryService


@pytest.fixture
def crm_settings(monkeypatch):
    monkeypatch.setattr(settings, "EVO_AI_CRM_URL", "http://crm.test")


def test_default_base_url_points_at_crm_internal_memory_namespace(crm_settings):
    # Regression test: HttpMemoryService() used to default to
    # settings.CORE_SERVICE_URL (the Go core service), which has no /memory/*
    # routes and 404s in production. The real memory endpoints live on the
    # Rails CRM under /api/v1/internal/memory/*, matching the pattern already
    # used by knowledge_preload.preload_knowledge for /api/v1/internal/knowledge/search.
    service = HttpMemoryService()

    assert service.base_url == "http://crm.test/api/v1/internal"


def test_explicit_base_url_is_not_overridden(crm_settings):
    service = HttpMemoryService(base_url="http://crm.test/api/v1")

    assert service.base_url == "http://crm.test/api/v1"
