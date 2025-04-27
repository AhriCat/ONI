import sympy as sp
import numpy as np
import re
import matplotlib.pyplot as plt

class Calculator:
    def __init__(self):
        self.symbols = {}
        self.operations = {
            'evaluate': self.evaluate_expression,
            'differentiate': self.differentiate,
            'integrate': self.integrate,
            'solve': self.solve_equation,
            'matrix_add': self.matrix_add,
            'matrix_subtract': self.matrix_subtract,
            'matrix_multiply': self.matrix_multiply,
            'determinant': self.determinant,
            'inverse': self.inverse,
            'eigenvalues': self.eigenvalues,
            'polynomial_roots': self.polynomial_roots,
            'graph': self.graph_function,
            'add': self.add,
            'subtract': self.subtract,
            'multiply': self.multiply,
            'divide': self.divide,
            'sin': self.sin,
            'cos': self.cos,
            'tan': self.tan,
            'log': self.log,
            'exp': self.exp,
            'power': self.power
        }

    def add_symbol(self, name):
        """Add a new symbol to the calculator."""
        symbol = sp.symbols(name)
        self.symbols[name] = symbol
        return symbol

    def evaluate_expression(self, expression):
        """Evaluate a symbolic expression."""
        expr = sp.sympify(expression)
        return expr.evalf()

    def differentiate(self, expression, symbol):
        """Differentiate a symbolic expression with respect to a given symbol."""
        expr = sp.sympify(expression)
        return sp.diff(expr, self.symbols[symbol])

    def integrate(self, expression, symbol):
        """Integrate a symbolic expression with respect to a given symbol."""
        expr = sp.sympify(expression)
        return sp.integrate(expr, self.symbols[symbol])

    def solve_equation(self, equation, symbol):
        """Solve an equation for a given symbol."""
        eq = sp.sympify(equation)
        return sp.solve(eq, self.symbols[symbol])

    def matrix_add(self, matrix_a, matrix_b):
        """Add two matrices."""
        return np.add(np.array(matrix_a), np.array(matrix_b))

    def matrix_subtract(self, matrix_a, matrix_b):
        """Subtract two matrices."""
        return np.subtract(np.array(matrix_a), np.array(matrix_b))

    def matrix_multiply(self, matrix_a, matrix_b):
        """Multiply two matrices."""
        return np.dot(np.array(matrix_a), np.array(matrix_b))

    def determinant(self, matrix):
        """Calculate the determinant of a matrix."""
        return np.linalg.det(np.array(matrix))

    def inverse(self, matrix):
        """Calculate the inverse of a matrix."""
        return np.linalg.inv(np.array(matrix))

    def eigenvalues(self, matrix):
        """Calculate the eigenvalues of a matrix."""
        return np.linalg.eigvals(np.array(matrix))

    def polynomial_roots(self, coefficients):
        """Find the roots of a polynomial given its coefficients."""
        return np.roots(coefficients)

    def graph_function(self, expression, symbol='x', start=-10, end=10):
        """Graph a function within a given range."""
        x = sp.symbols(symbol)
        expr = sp.sympify(expression)
        f = sp.lambdify(x, expr, 'numpy')

        x_vals = np.linspace(start, end, 400)
        y_vals = f(x_vals)

        plt.plot(x_vals, y_vals)
        plt.xlabel(symbol)
        plt.ylabel('f({})'.format(symbol))
        plt.title('Graph of {}'.format(expression))
        plt.grid(True)
        plt.show()

    # Basic arithmetic operations
    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            return "Error: Division by zero"
        return a / b

    # Trigonometric functions
    def sin(self, angle):
        return np.sin(np.radians(angle))

    def cos(self, angle):
        return np.cos(np.radians(angle))

    def tan(self, angle):
        return np.tan(np.radians(angle))

    # Logarithm and exponential
    def log(self, value, base=np.e):
        if value <= 0:
            return "Error: Logarithm of non-positive number"
        return np.log(value) / np.log(base)

    def exp(self, value):
        return np.exp(value)

    # Power function
    def power(self, base, exponent):
        return np.power(base, exponent)

    def parse_input(self, input_string):
        """Parse the input string and return the operation type and parameters."""
        input_string = input_string.lower().strip()
        
        # Check for matrix operations
        matrix_ops = re.match(r'(add|subtract|multiply) matrices (.*) and (.*)', input_string)
        if matrix_ops:
            op, mat_a, mat_b = matrix_ops.groups()
            return f'matrix_{op}', {'matrix_a': eval(mat_a), 'matrix_b': eval(mat_b)}
        
        # Check for determinant, inverse, eigenvalues
        matrix_single_op = re.match(r'(determinant|inverse|eigenvalues) of matrix (.*)', input_string)
        if matrix_single_op:
            op, mat = matrix_single_op.groups()
            return op, {'matrix': eval(mat)}
        
        # Check for polynomial roots
        poly_roots = re.match(r'roots of polynomial (.*)', input_string)
        if poly_roots:
            coeffs = poly_roots.group(1)
            return 'polynomial_roots', {'coefficients': eval(coeffs)}
        
        # Check for differentiation and integration
        diff_int = re.match(r'(differentiate|integrate) (.*) with respect to (\w+)', input_string)
        if diff_int:
            op, expr, symbol = diff_int.groups()
            self.add_symbol(symbol)
            return op, {'expression': expr, 'symbol': symbol}
        
        # Check for equation solving
        solve_eq = re.match(r'solve (.*) for (\w+)', input_string)
        if solve_eq:
            eq, symbol = solve_eq.groups()
            self.add_symbol(symbol)
            return 'solve', {'equation': eq, 'symbol': symbol}
        
        # Check for graphing
        graph = re.match(r'graph (.*) from (-?\d+) to (-?\d+)', input_string)
        if graph:
            expr, start, end = graph.groups()
            return 'graph', {'expression': expr, 'start': int(start), 'end': int(end)}
        
        # Basic arithmetic operations
        basic_ops = re.match(r'(add|subtract|multiply|divide) (-?\d+(\.\d+)?) and (-?\d+(\.\d+)?)', input_string)
        if basic_ops:
            op, a, _, b, _ = basic_ops.groups()
            return op, {'a': float(a), 'b': float(b)}
        
        # Trigonometric functions
        trig_ops = re.match(r'(sin|cos|tan) of (-?\d+(\.\d+)?)', input_string)
        if trig_ops:
            op, angle, _ = trig_ops.groups()
            return op, {'angle': float(angle)}
        
        # Logarithm
        log_op = re.match(r'log of (-?\d+(\.\d+)?)(?: base (-?\d+(\.\d+)?))?', input_string)
        if log_op:
            value, _, base, _ = log_op.groups()
            if base:
                return 'log', {'value': float(value), 'base': float(base)}
            return 'log', {'value': float(value)}
        
        # Exponential and power
        exp_op = re.match(r'exp of (-?\d+(\.\d+)?)', input_string)
        if exp_op:
            value, _ = exp_op.groups()
            return 'exp', {'value': float(value)}
        
        power_op = re.match(r'(-?\d+(\.\d+)?) to the power of (-?\d+(\.\d+)?)', input_string)
        if power_op:
            base, _, exponent, _ = power_op.groups()
            return 'power', {'base': float(base), 'exponent': float(exponent)}
        
        # If no specific operation is detected, assume it's an expression to evaluate
        return 'evaluate', {'expression': input_string}

    def calculate(self, input_string):
        """Parse the input string, determine the operation, and perform the calculation."""
        try:
            operation, params = self.parse_input(input_string)
            
            if operation in self.operations:
                return self.operations[operation](**params)
            else:
                raise ValueError(f"Operation '{operation}' is not recognized.")
        except Exception as e:
            return f"Error: {str(e)}"

# Example usage:
calculator = Calculator()

# Examples of using the calculate method with string inputs
print(calculator.calculate("2*x + 3*y - 5"))
print(calculator.calculate("differentiate x**2 + y**2 with respect to x"))
print(calculator.calculate("solve x**2 - 4 for x"))
print(calculator.calculate("add matrices [[1, 2], [3, 4]] and [[5, 6], [7, 8]]"))
print(calculator.calculate("determinant of matrix [[1, 2], [3, 4]]"))
print(calculator.calculate("roots of polynomial [1, -3, 2]"))
print(calculator.calculate("graph x**2 - 4 from -10 to 10"))
print(calculator.calculate("add 5 and 10"))
print(calculator.calculate("sin of 30"))
print(calculator.calculate("log of 100 base 10"))
