def test_max_internal_within() -> None:
    from reblock.budget import Curve
    from scripts.calibrate_joint_target import max_internal_within
    ext = Curve([0, 1, 2, 3], [0.5, 0.72, 0.8, 0.9])
    inte = Curve([0, 1, 2, 3], [0.1, 0.30, 0.45, 0.50])
    disp = Curve([0, 1, 2, 3], [0.1, 0.2, 0.40, 0.60])
    # samples with ext>=.70 and disp<=.45: i=1 (int .30), i=2 (int .45); i=3 disp .60 excluded
    assert max_internal_within(ext, inte, disp, e_min=0.70, d_max=0.45) == 0.45
    assert max_internal_within(ext, inte, disp, e_min=0.99, d_max=0.45) == float("-inf")
