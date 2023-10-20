import casadi as ca

# Define symbolic variables
x = ca.MX.sym('x')  # Symbolic variable
condition = (x > 0)  # Condition for the if-else statement

# Define expressions for if and else cases
x_if_true = x * 2
x_if_false = x / 2

# Create if-else statement using if_else function
result = ca.if_else(condition, x_if_true, x_if_false)

# Example usage
input_value = 3  # Value of x
output_value = result.expand()(input_value)  # Evaluate the expression

print("Result:", output_value)
