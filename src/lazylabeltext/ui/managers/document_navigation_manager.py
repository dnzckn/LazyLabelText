"""Document navigation manager."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lazylabeltext.ui.main_window import MainWindow


class DocumentNavigationManager:
    """Manages document selection and navigation."""

    def __init__(self, main_window: MainWindow) -> None:
        self.mw = main_window

    def select_next(self) -> None:
        """Select the next document in the tree."""
        tree = self.mw.left_panel.tree
        current = tree.currentItem()
        if current is None:
            if tree.topLevelItemCount() > 0:
                tree.setCurrentItem(tree.topLevelItem(0))
            return

        idx = tree.indexOfTopLevelItem(current)
        if idx < tree.topLevelItemCount() - 1:
            tree.setCurrentItem(tree.topLevelItem(idx + 1))

    def select_previous(self) -> None:
        """Select the previous document in the tree."""
        tree = self.mw.left_panel.tree
        current = tree.currentItem()
        if current is None:
            return

        idx = tree.indexOfTopLevelItem(current)
        if idx > 0:
            tree.setCurrentItem(tree.topLevelItem(idx - 1))
