"""The sign guide pane (src/ui/guide.py) on the offscreen Qt platform: entries, copy, photos and reflow."""

import pytest

from src import config
from src.ui import guide
from src.ui.guide import GUIDE_ENTRIES, GUIDE_INTRO, GuidePane, columns_for

# Tall enough that Fusion's scroll bar (offscreen; not an overlay like macOS's)
# never appears, so the viewport is as wide as the pane.
TALL = 5000


@pytest.fixture
def pane(qapp):
    widget = GuidePane()
    yield widget
    widget.close()


def _position(pane, tile):
    """(row, column) of ``tile`` in its section grid."""
    for grid, tiles in pane._sections:
        if tile in tiles:
            row, column, _, _ = grid.getItemPosition(grid.indexOf(tile))
            return row, column
    raise AssertionError(f"{tile.entry.name} is in no grid")


def test_entries_follow_the_class_order():
    assert [entry.name for entry in GUIDE_ENTRIES] == config.CLASSES
    assert GUIDE_ENTRIES[0].image_path == config.GUIDE_DIR / "A.jpg"


def test_intro_describes_how_commits_work():
    hold, still = GUIDE_INTRO
    for phrase in ("Hold a sign", "lower your hand", "NOTHING", "different letter"):
        assert phrase in hold
    assert "J and Z are read as still poses" in still


def test_notes_are_exact_and_only_on_five_signs():
    notes = {entry.name: entry.note for entry in GUIDE_ENTRIES}
    assert notes.pop("J") == "Still pose only"
    assert notes.pop("Z") == "Still pose only"
    assert notes.pop("SPACE") == "Adds a space"
    assert notes.pop("DELETE") == "Deletes the last letter"
    assert notes.pop("NOTHING") == "No hand in view"
    assert set(notes.values()) == {None}


def test_photos_are_rendered_once_at_2x(qapp):
    pixmap = guide.rounded_photo(GUIDE_ENTRIES[0].image_path)
    assert (pixmap.width(), pixmap.height(), pixmap.devicePixelRatio()) == (176, 176, 2.0)
    corner = pixmap.toImage().pixelColor(0, 0)
    assert corner.alpha() == 0  # the rounded corner is clipped away
    assert guide.rounded_photo(config.GUIDE_DIR / "missing.jpg").isNull()


def test_every_tile_has_its_photo_and_accessible_name(pane):
    assert [tile.entry for tile in pane.tiles] == list(GUIDE_ENTRIES)
    for tile in pane.tiles:
        pixmap = tile.image.pixmap()
        assert not pixmap.isNull(), tile.entry.name
        size = pixmap.deviceIndependentSize()  # QLabel hands back the pixmap at the screen's ratio
        assert (size.width(), size.height()) == (88, 88)
        assert tile.name_label.text() == tile.entry.name
        assert (tile.note_label is None) == (tile.entry.note is None)
        expected = "NOTHING: empty view" if tile.entry.name == "NOTHING" else f"{tile.entry.name} hand shape"
        assert tile.image.accessibleName() == expected


def test_columns_for_widths():
    assert [columns_for(w) for w in (320, 236, 200)] == [3, 2, 1]


def test_grid_reflows_when_the_pane_narrows(pane, qapp):
    pane.resize(320, TALL)
    pane.show()
    qapp.processEvents()
    assert pane.viewport().width() == 320
    assert pane.columns == 3
    tile_d = pane.tiles[config.CLASSES.index("D")]
    assert _position(pane, tile_d) == (1, 0)

    pane.resize(guide.MIN_WIDTH, TALL)
    qapp.processEvents()
    assert pane.columns == 2
    assert _position(pane, tile_d) == (1, 1)
    assert _position(pane, pane.tiles[config.CLASSES.index("NOTHING")]) == (1, 0)


def test_missing_photos_still_build_every_tile(qapp, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "GUIDE_DIR", tmp_path)
    pane = GuidePane()
    try:
        assert len(pane.tiles) == 29
        for tile in pane.tiles:
            assert tile.image.pixmap().isNull()
            assert (tile.image.width(), tile.image.height()) == (88, 88)
            assert tile.name_label.text() == tile.entry.name
    finally:
        pane.close()


def test_credit_line_names_the_photo_source(pane):
    assert pane.credit_label.text() == "Photos: Dan Rasband, ASL Alphabet Test (CC0)"
