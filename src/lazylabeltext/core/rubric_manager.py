"""Rubric management with versioning."""

from __future__ import annotations

import json
import logging

from lazylabeltext.core.database import Database
from lazylabeltext.core.exceptions import RubricValidationError
from lazylabeltext.core.models import Category, Rubric

logger = logging.getLogger("lazylabeltext")


class RubricManager:
    """Manages rubric CRUD operations and versioning."""

    def __init__(self, database: Database) -> None:
        self.db = database

    def create_rubric(self, name: str, categories: list[Category]) -> Rubric:
        """Create a new rubric (version 1)."""
        self._validate_categories(categories)
        rubric = Rubric(name=name, version=1, categories=categories)
        rubric.id = self.db.insert_rubric(rubric)
        logger.info("Created rubric '%s' v1 with %d categories", name, len(categories))
        return rubric

    def update_rubric(self, rubric_id: int, categories: list[Category]) -> Rubric:
        """Create a new version of an existing rubric."""
        existing = self.db.get_rubric(rubric_id)
        if existing is None:
            raise RubricValidationError(f"Rubric {rubric_id} not found")

        self._validate_categories(categories)
        new_version = Rubric(
            name=existing.name,
            version=existing.version + 1,
            categories=categories,
        )
        new_version.id = self.db.insert_rubric(new_version)
        logger.info("Updated rubric '%s' to v%d", existing.name, new_version.version)
        return new_version

    def get_active_rubric(self) -> Rubric | None:
        """Get the latest rubric version."""
        return self.db.get_latest_rubric()

    def get_rubric(self, rubric_id: int) -> Rubric | None:
        return self.db.get_rubric(rubric_id)

    def list_versions(self) -> list[Rubric]:
        return self.db.get_all_rubric_versions()

    def import_from_json(self, json_str: str, name: str = "Imported") -> Rubric:
        """Import a rubric from a JSON string."""
        categories = self.parse_categories_json(json_str)
        return self.create_rubric(name, categories)

    def export_to_json(self, rubric_id: int) -> str:
        """Export a rubric to JSON string."""
        rubric = self.db.get_rubric(rubric_id)
        if rubric is None:
            raise RubricValidationError(f"Rubric {rubric_id} not found")
        data = {
            "name": rubric.name,
            "version": rubric.version,
            "categories": [
                {
                    "name": c.name,
                    "definition": c.definition,
                    "exemplars": c.exemplars,
                    "boundary_cases": c.boundary_cases,
                    "confidence_threshold": c.confidence_threshold,
                }
                for c in rubric.categories
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

    def parse_categories_json(self, json_str: str) -> list[Category]:
        """Parse a JSON string into a list of Category objects."""
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise RubricValidationError(f"Invalid JSON: {e}") from e

        if isinstance(data, dict):
            cats_data = data.get("categories", [])
        elif isinstance(data, list):
            cats_data = data
        else:
            raise RubricValidationError("JSON must be an object or array")

        categories = []
        for item in cats_data:
            if not isinstance(item, dict):
                raise RubricValidationError("Each category must be an object")
            if "name" not in item:
                raise RubricValidationError("Each category must have a 'name'")
            categories.append(
                Category(
                    name=item["name"],
                    definition=item.get("definition", ""),
                    exemplars=item.get("exemplars", []),
                    boundary_cases=item.get("boundary_cases", []),
                    confidence_threshold=item.get("confidence_threshold", 0.85),
                    parent=item.get("parent"),
                )
            )
        return categories

    def _validate_categories(self, categories: list[Category]) -> None:
        """Validate category list."""
        if not categories:
            raise RubricValidationError("Rubric must have at least one category")

        names = [c.name for c in categories]
        if len(names) != len(set(names)):
            raise RubricValidationError("Category names must be unique")

        for cat in categories:
            if not cat.name.strip():
                raise RubricValidationError("Category name cannot be empty")
