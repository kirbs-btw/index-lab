use index_convergence::ConvergenceConfig;

#[test]
fn test_default_config() {
    let config = ConvergenceConfig::default();
    assert!(config.validate().is_ok());
}
