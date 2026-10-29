"""The migrations produce exactly the schema the models describe (like `alembic check`)."""

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine

from app.db.models import Base

pytestmark = pytest.mark.db


def test_models_match_migrations(test_db_settings):
    engine = create_engine(test_db_settings.database_url())
    try:
        with engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    finally:
        engine.dispose()
    assert diff == []
