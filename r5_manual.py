# Dictionary of items and their expected returns
expected_returns = {
    "Haystacks": -0.0048,
    "Ranch_sauce": -0.0072,
    "Cacti_Needle": -0.412,
    "Solar_panels": -0.089,
    "Red_Flags": 0.509,
    "VR_Monocle": 0.224,
    "Quantum_Coffee": -0.6679,
    "Moonshine": 0.03,
    "Striped_shirts": 0.0021,
}

actual_returns = {
    "Haystacks": -0.0048,
    "Ranch_sauce": -0.0072,
    "Cacti_Needle": -0.412,
    "Solar_panels": -0.089,
    "Red_Flags": 0.509,
    "VR_Monocle": 0.224,
    "Quantum_Coffee": -0.6679,
    "Moonshine": 0.03,
    "Striped_shirts": 0.0021,
}

def calculate_borrowing_fee(x):
    """Calculate borrowing fee for amount x"""
    return (x**2)/((250/3)*10000)

def find_optimal_amount(return_rate):
    """
    Find optimal amount for a given return rate
    Marginal return = return_rate
    Marginal cost = 2x/((250/3)*10000)
    At optimal point: return_rate = 2x/((250/3)*10000)
    Solving for x: x = return_rate * ((250/3)*10000)/2
    """
    # Calculate theoretical optimal amount
    optimal = abs(return_rate) * ((250/3)*10000)/2
    
    # Round to nearest 10000
    rounded = round(optimal/10000) * 10000
    
    # Return negative amount if return rate is negative (for shorting)
    return -rounded if return_rate < 0 else rounded

print("Allocations based on expected returns:")
total_investment = 0
allocations = {}
expected_pnl = 0

for asset, rate in expected_returns.items():
    amount = find_optimal_amount(rate)
    if True:
        allocations[asset] = amount
        total_investment += abs(amount)
        expected_return = amount * rate - calculate_borrowing_fee(abs(amount))
        print(f"{asset}: ${amount:,} (Expected return: ${expected_return:,.2f})")
        expected_pnl += expected_return

print(f"\nTotal investment: ${total_investment:,}")
print(f"Expected total P&L: ${expected_pnl:,.2f}")

# Calculate actual P&L using the same allocations
actual_pnl = 0
print("\nActual returns:")
for asset, amount in allocations.items():
    actual_return = amount * actual_returns[asset] - calculate_borrowing_fee(abs(amount))
    print(f"{asset}: ${actual_return:,.2f}")
    actual_pnl += actual_return

print(f"\nActual total P&L: ${actual_pnl:,.2f}")