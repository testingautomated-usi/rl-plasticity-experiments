from statsmodels.stats.power import TTestIndPower


def parametric_power_analysis(effect: float = 0.8, alpha: float = 0.05, power: float = 0.8) -> float:
    analysis = TTestIndPower()
    return analysis.solve_power(effect, power=power, alpha=alpha)
