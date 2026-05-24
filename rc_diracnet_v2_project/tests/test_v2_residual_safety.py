from rc_diracnet_v2.training.stage_gate import residual_safety_verdict


def test_residual_safety_flags_large_delta():
    v=residual_safety_verdict({"physics_gate_pass":True,"delta_ratio_median":0.8,"delta_abs_max_mev":4.9,"delta_cap_mev":5,"loo_rms_mev":500,"e_orb_only_mae_mev":100})
    assert v == "CHEATING_RISK"
