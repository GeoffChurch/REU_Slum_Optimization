from reblock.method_labels import friendly_method_name


def test_friendly_method_name_maps_known_keys() -> None:
    assert friendly_method_name("osm_footpaths") == "OSM Footpaths"
    assert friendly_method_name("clearance_looped") == "Looped Tree"
    assert friendly_method_name("euclidean_grid") == "Grid"
    assert friendly_method_name("greedy_arterial_repulsion") == "Throughways"


def test_friendly_method_name_falls_back_to_raw_key_for_unmapped_method() -> None:
    # A key that is deliberately not a method, so labelling a real one cannot break this the way
    # mapping `topology` did.
    assert friendly_method_name("not_a_registered_method") == "not_a_registered_method"
