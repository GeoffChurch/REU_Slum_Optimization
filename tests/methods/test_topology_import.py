def test_topology_public_api() -> None:
    import topology

    for name in ("MyGraph", "graphFromMyFaces", "build_all_roads", "k_complexity"):
        assert hasattr(topology, name)
