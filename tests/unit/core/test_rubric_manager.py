"""Tests for rubric manager."""

from __future__ import annotations

import pytest

from lazylabeltext.core.exceptions import RubricValidationError
from lazylabeltext.core.models import Category
from lazylabeltext.core.rubric_manager import RubricManager


class TestRubricManager:
    @pytest.fixture
    def manager(self, sample_database):
        return RubricManager(sample_database)

    def test_create_rubric(self, manager):
        cats = [Category(name="cat1", definition="def1")]
        rubric = manager.create_rubric("Test", cats)
        assert rubric.id is not None
        assert rubric.version == 1
        assert len(rubric.categories) == 1

    def test_update_rubric_creates_new_version(self, manager):
        cats = [Category(name="cat1", definition="def1")]
        r1 = manager.create_rubric("Test", cats)

        cats2 = [Category(name="cat1", definition="updated"), Category(name="cat2")]
        r2 = manager.update_rubric(r1.id, cats2)

        assert r2.version == 2
        assert len(r2.categories) == 2

    def test_empty_categories_raises(self, manager):
        with pytest.raises(RubricValidationError):
            manager.create_rubric("Empty", [])

    def test_duplicate_names_raises(self, manager):
        with pytest.raises(RubricValidationError):
            manager.create_rubric("Dup", [Category(name="a"), Category(name="a")])

    def test_import_from_json(self, manager, sample_rubric_json):
        rubric = manager.import_from_json(sample_rubric_json)
        assert len(rubric.categories) == 2
        assert rubric.categories[0].name == "procedure"

    def test_export_to_json(self, manager):
        cats = [Category(name="test", definition="Testing category")]
        r = manager.create_rubric("Export", cats)
        json_str = manager.export_to_json(r.id)
        assert '"test"' in json_str
        assert '"Testing category"' in json_str

    def test_parse_invalid_json_raises(self, manager):
        with pytest.raises(RubricValidationError):
            manager.parse_categories_json("not json")
