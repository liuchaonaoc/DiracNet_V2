def test_import_v1_ports():
    from rc_diracnet_v2.utils.grid import RadialGrid
    from rc_diracnet_v2.physics.dirac_operator import DiracRadialOperator
    from rc_diracnet_v2.encoders.quantum_encoder import GlobalQuantumEncoder
    from rc_diracnet_v2.losses.pde_loss import DiracPDELoss
    assert RadialGrid and DiracRadialOperator and GlobalQuantumEncoder and DiracPDELoss
