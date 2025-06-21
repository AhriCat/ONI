import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as optimize
import sympy as sp
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
import io
import base64
from datetime import datetime
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# App metadata
APP_NAME = "oni_science_lab"
APP_DESCRIPTION = "Scientific computing and analysis tools for ONI"
APP_VERSION = "1.0.0"
APP_AUTHOR = "ONI Team"
APP_CATEGORY = "Science"
APP_DEPENDENCIES = ["numpy", "pandas", "matplotlib", "scipy", "sympy"]
APP_DEFAULT = False

class ONIScienceLab:
    """
    ONI Science Lab - Scientific computing and analysis tools.
    
    Provides capabilities for:
    - Statistical analysis
    - Data visualization
    - Symbolic mathematics
    - Curve fitting
    - Hypothesis testing
    - Numerical simulations
    """
    
    def __init__(self):
        """Initialize the ONI Science Lab."""
        self.datasets = {}
        self.analysis_results = {}
        self.symbolic_variables = {}
        self.simulation_results = {}
        self.figure_counter = 0
        
        # Initialize symbolic variables
        self._init_symbolic_variables()
        
        logger.info("ONI Science Lab initialized")
    
    def _init_symbolic_variables(self):
        """Initialize common symbolic variables."""
        self.symbolic_variables = {
            'x': sp.Symbol('x'),
            'y': sp.Symbol('y'),
            'z': sp.Symbol('z'),
            't': sp.Symbol('t'),
            'a': sp.Symbol('a'),
            'b': sp.Symbol('b'),
            'c': sp.Symbol('c'),
            'n': sp.Symbol('n')
        }
    
    def load_dataset(self, data: Union[str, Dict, List, pd.DataFrame], name: str = None) -> pd.DataFrame:
        """
        Load a dataset into the science lab.
        
        Args:
            data: Dataset as string (CSV/JSON), dictionary, list, or DataFrame
            name: Name to assign to the dataset
            
        Returns:
            pandas.DataFrame: The loaded dataset
        """
        try:
            # Generate a name if not provided
            if name is None:
                name = f"dataset_{len(self.datasets) + 1}"
            
            # Convert data to DataFrame
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, str):
                # Try parsing as CSV
                try:
                    df = pd.read_csv(io.StringIO(data))
                except:
                    # Try parsing as JSON
                    try:
                        df = pd.read_json(io.StringIO(data))
                    except:
                        raise ValueError("Could not parse string as CSV or JSON")
            elif isinstance(data, dict):
                df = pd.DataFrame(data)
            elif isinstance(data, list):
                if all(isinstance(item, dict) for item in data):
                    df = pd.DataFrame(data)
                else:
                    df = pd.DataFrame(data)
            else:
                raise ValueError("Unsupported data type")
            
            # Store the dataset
            self.datasets[name] = df
            
            logger.info(f"Loaded dataset '{name}' with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def get_dataset(self, name: str) -> pd.DataFrame:
        """
        Get a dataset by name.
        
        Args:
            name: Name of the dataset
            
        Returns:
            pandas.DataFrame: The dataset
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        return self.datasets[name]
    
    def list_datasets(self) -> Dict[str, Tuple[int, int]]:
        """
        List all available datasets.
        
        Returns:
            Dict[str, Tuple[int, int]]: Dictionary mapping dataset names to their shapes
        """
        return {name: df.shape for name, df in self.datasets.items()}
    
    def describe_dataset(self, name: str) -> Dict[str, Any]:
        """
        Get descriptive statistics for a dataset.
        
        Args:
            name: Name of the dataset
            
        Returns:
            Dict[str, Any]: Descriptive statistics
        """
        if name not in self.datasets:
            raise ValueError(f"Dataset '{name}' not found")
        
        df = self.datasets[name]
        
        # Get basic statistics
        stats_df = df.describe(include='all')
        
        # Convert to dictionary
        stats_dict = stats_df.to_dict()
        
        # Add additional information
        stats_dict['_info'] = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict()
        }
        
        # Store the results
        result_name = f"{name}_stats"
        self.analysis_results[result_name] = stats_dict
        
        return stats_dict
    
    def plot_data(self, 
                 x: Union[str, List, np.ndarray], 
                 y: Union[str, List, np.ndarray] = None, 
                 dataset: str = None,
                 plot_type: str = 'line',
                 title: str = None,
                 xlabel: str = None,
                 ylabel: str = None,
                 figsize: Tuple[int, int] = (10, 6),
                 return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Create a plot of data.
        
        Args:
            x: x-values or column name
            y: y-values or column name (optional for histograms)
            dataset: Name of the dataset to use (optional)
            plot_type: Type of plot ('line', 'scatter', 'bar', 'hist', 'box', 'pie')
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Get data
            x_data = x
            y_data = y
            
            if dataset is not None:
                if dataset not in self.datasets:
                    raise ValueError(f"Dataset '{dataset}' not found")
                
                df = self.datasets[dataset]
                
                if isinstance(x, str):
                    if x not in df.columns:
                        raise ValueError(f"Column '{x}' not found in dataset '{dataset}'")
                    x_data = df[x]
                
                if y is not None and isinstance(y, str):
                    if y not in df.columns:
                        raise ValueError(f"Column '{y}' not found in dataset '{dataset}'")
                    y_data = df[y]
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create plot based on type
            if plot_type == 'line':
                ax.plot(x_data, y_data)
            elif plot_type == 'scatter':
                ax.scatter(x_data, y_data)
            elif plot_type == 'bar':
                ax.bar(x_data, y_data)
            elif plot_type == 'hist':
                ax.hist(x_data, bins=30)
            elif plot_type == 'box':
                if dataset is not None and isinstance(x, str):
                    df[x].plot(kind='box', ax=ax)
                else:
                    ax.boxplot(x_data)
            elif plot_type == 'pie':
                ax.pie(y_data, labels=x_data, autopct='%1.1f%%')
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Set labels and title
            if title:
                ax.set_title(title)
            if xlabel:
                ax.set_xlabel(xlabel)
            if ylabel:
                ax.set_ylabel(ylabel)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error creating plot: {e}")
            raise
    
    def perform_regression(self, 
                          x: Union[str, List, np.ndarray], 
                          y: Union[str, List, np.ndarray],
                          dataset: str = None,
                          reg_type: str = 'linear',
                          degree: int = 2,
                          plot_result: bool = True,
                          return_base64: bool = True) -> Dict[str, Any]:
        """
        Perform regression analysis.
        
        Args:
            x: x-values or column name
            y: y-values or column name
            dataset: Name of the dataset to use (optional)
            reg_type: Type of regression ('linear', 'polynomial', 'exponential', 'logarithmic')
            degree: Degree of polynomial (for polynomial regression)
            plot_result: Whether to create a plot of the regression
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Dict[str, Any]: Regression results
        """
        try:
            # Get data
            x_data = x
            y_data = y
            
            if dataset is not None:
                if dataset not in self.datasets:
                    raise ValueError(f"Dataset '{dataset}' not found")
                
                df = self.datasets[dataset]
                
                if isinstance(x, str):
                    if x not in df.columns:
                        raise ValueError(f"Column '{x}' not found in dataset '{dataset}'")
                    x_data = df[x]
                
                if isinstance(y, str):
                    if y not in df.columns:
                        raise ValueError(f"Column '{y}' not found in dataset '{dataset}'")
                    y_data = df[y]
            
            # Convert to numpy arrays
            x_array = np.array(x_data, dtype=float)
            y_array = np.array(y_data, dtype=float)
            
            # Perform regression
            result = {}
            
            if reg_type == 'linear':
                # Linear regression: y = mx + b
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_array, y_array)
                
                result = {
                    'type': 'linear',
                    'equation': f'y = {slope:.6f}x + {intercept:.6f}',
                    'parameters': {
                        'slope': slope,
                        'intercept': intercept
                    },
                    'statistics': {
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err
                    },
                    'predict': lambda x_new: slope * x_new + intercept
                }
                
                # Create prediction function for plotting
                def predict_fn(x_new):
                    return slope * x_new + intercept
                
            elif reg_type == 'polynomial':
                # Polynomial regression: y = a_n * x^n + ... + a_1 * x + a_0
                coeffs = np.polyfit(x_array, y_array, degree)
                poly = np.poly1d(coeffs)
                
                # Calculate R-squared
                y_pred = poly(x_array)
                ss_total = np.sum((y_array - np.mean(y_array))**2)
                ss_residual = np.sum((y_array - y_pred)**2)
                r_squared = 1 - (ss_residual / ss_total)
                
                # Create equation string
                equation = "y = "
                for i, coeff in enumerate(coeffs):
                    power = degree - i
                    if power > 1:
                        equation += f"{coeff:.6f}x^{power} + "
                    elif power == 1:
                        equation += f"{coeff:.6f}x + "
                    else:
                        equation += f"{coeff:.6f}"
                
                result = {
                    'type': 'polynomial',
                    'degree': degree,
                    'equation': equation,
                    'parameters': {
                        'coefficients': coeffs.tolist()
                    },
                    'statistics': {
                        'r_squared': r_squared
                    },
                    'predict': lambda x_new: poly(x_new)
                }
                
                # Create prediction function for plotting
                predict_fn = poly
                
            elif reg_type == 'exponential':
                # Exponential regression: y = a * exp(b * x)
                # Linearize: ln(y) = ln(a) + b * x
                valid_indices = y_array > 0  # Can't take log of negative or zero values
                if not np.all(valid_indices):
                    logger.warning("Some y-values are <= 0, removing for exponential regression")
                
                x_valid = x_array[valid_indices]
                y_valid = y_array[valid_indices]
                
                if len(x_valid) < 2:
                    raise ValueError("Not enough valid data points for exponential regression")
                
                # Linearize
                y_log = np.log(y_valid)
                
                # Linear regression on linearized data
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_log)
                
                # Convert back to exponential form
                a = np.exp(intercept)
                b = slope
                
                result = {
                    'type': 'exponential',
                    'equation': f'y = {a:.6f} * exp({b:.6f} * x)',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'statistics': {
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err
                    },
                    'predict': lambda x_new: a * np.exp(b * x_new)
                }
                
                # Create prediction function for plotting
                def predict_fn(x_new):
                    return a * np.exp(b * x_new)
                
            elif reg_type == 'logarithmic':
                # Logarithmic regression: y = a + b * ln(x)
                valid_indices = x_array > 0  # Can't take log of negative or zero values
                if not np.all(valid_indices):
                    logger.warning("Some x-values are <= 0, removing for logarithmic regression")
                
                x_valid = x_array[valid_indices]
                y_valid = y_array[valid_indices]
                
                if len(x_valid) < 2:
                    raise ValueError("Not enough valid data points for logarithmic regression")
                
                # Transform x
                x_log = np.log(x_valid)
                
                # Linear regression on transformed data
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_valid)
                
                # Parameters
                a = intercept
                b = slope
                
                result = {
                    'type': 'logarithmic',
                    'equation': f'y = {a:.6f} + {b:.6f} * ln(x)',
                    'parameters': {
                        'a': a,
                        'b': b
                    },
                    'statistics': {
                        'r_squared': r_value**2,
                        'p_value': p_value,
                        'std_err': std_err
                    },
                    'predict': lambda x_new: a + b * np.log(x_new)
                }
                
                # Create prediction function for plotting
                def predict_fn(x_new):
                    return a + b * np.log(x_new)
                
            else:
                raise ValueError(f"Unsupported regression type: {reg_type}")
            
            # Create plot if requested
            if plot_result:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Plot original data
                ax.scatter(x_array, y_array, color='blue', alpha=0.6, label='Data')
                
                # Plot regression line
                x_range = np.linspace(min(x_array), max(x_array), 100)
                
                if reg_type == 'logarithmic' or reg_type == 'exponential':
                    # Filter out invalid values for prediction
                    if reg_type == 'logarithmic':
                        x_range = x_range[x_range > 0]
                    
                    # Skip prediction if no valid values
                    if len(x_range) > 0:
                        y_pred = predict_fn(x_range)
                        ax.plot(x_range, y_pred, color='red', label=f'Regression ({result["equation"]})')
                else:
                    y_pred = predict_fn(x_range)
                    ax.plot(x_range, y_pred, color='red', label=f'Regression ({result["equation"]})')
                
                # Add labels and title
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(f'{reg_type.capitalize()} Regression')
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
                # Tight layout
                fig.tight_layout()
                
                # Convert to base64 if requested
                if return_base64:
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=100)
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    plt.close(fig)
                    result['plot'] = img_str
                else:
                    result['figure'] = fig
            
            # Store the results
            result_name = f"{dataset or 'custom'}_{reg_type}_regression"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing regression: {e}")
            raise
    
    def perform_hypothesis_test(self,
                               data1: Union[str, List, np.ndarray],
                               data2: Union[str, List, np.ndarray] = None,
                               dataset: str = None,
                               test_type: str = 'ttest',
                               alpha: float = 0.05) -> Dict[str, Any]:
        """
        Perform a hypothesis test.
        
        Args:
            data1: First data sample or column name
            data2: Second data sample or column name (optional for some tests)
            dataset: Name of the dataset to use (optional)
            test_type: Type of test ('ttest', 'anova', 'chi2', 'correlation')
            alpha: Significance level
            
        Returns:
            Dict[str, Any]: Hypothesis test results
        """
        try:
            # Get data
            data1_values = data1
            data2_values = data2
            
            if dataset is not None:
                if dataset not in self.datasets:
                    raise ValueError(f"Dataset '{dataset}' not found")
                
                df = self.datasets[dataset]
                
                if isinstance(data1, str):
                    if data1 not in df.columns:
                        raise ValueError(f"Column '{data1}' not found in dataset '{dataset}'")
                    data1_values = df[data1]
                
                if data2 is not None and isinstance(data2, str):
                    if data2 not in df.columns:
                        raise ValueError(f"Column '{data2}' not found in dataset '{dataset}'")
                    data2_values = df[data2]
            
            # Convert to numpy arrays
            data1_array = np.array(data1_values, dtype=float)
            
            if data2_values is not None:
                data2_array = np.array(data2_values, dtype=float)
            
            # Perform hypothesis test
            result = {}
            
            if test_type == 'ttest':
                # t-test
                if data2_values is None:
                    # One-sample t-test (against mean=0)
                    t_stat, p_value = stats.ttest_1samp(data1_array, 0)
                    test_name = "One-sample t-test"
                else:
                    # Two-sample t-test
                    t_stat, p_value = stats.ttest_ind(data1_array, data2_array)
                    test_name = "Two-sample t-test"
                
                result = {
                    'type': test_type,
                    'name': test_name,
                    'statistic': t_stat,
                    'p_value': p_value,
                    'alpha': alpha,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at alpha={alpha}"
                }
                
            elif test_type == 'anova':
                # ANOVA
                if data2_values is None:
                    raise ValueError("ANOVA requires at least two data samples")
                
                # Convert inputs to list of arrays for ANOVA
                samples = [data1_array]
                if isinstance(data2_array, list) and all(isinstance(x, np.ndarray) for x in data2_array):
                    samples.extend(data2_array)
                else:
                    samples.append(data2_array)
                
                f_stat, p_value = stats.f_oneway(*samples)
                
                result = {
                    'type': test_type,
                    'name': "One-way ANOVA",
                    'statistic': f_stat,
                    'p_value': p_value,
                    'alpha': alpha,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at alpha={alpha}"
                }
                
            elif test_type == 'chi2':
                # Chi-square test
                if data2_values is None:
                    # Goodness of fit test
                    observed = data1_array
                    expected = np.ones_like(observed) * np.mean(observed)
                    chi2_stat, p_value = stats.chisquare(observed, expected)
                    test_name = "Chi-square goodness of fit test"
                else:
                    # Independence test
                    # Reshape data into contingency table if needed
                    if len(data1_array.shape) == 1 and len(data2_array.shape) == 1:
                        # Create contingency table from two categorical variables
                        contingency = pd.crosstab(data1_array, data2_array).values
                    else:
                        # Assume data is already in contingency table format
                        contingency = np.array([data1_array, data2_array])
                    
                    chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency)
                    test_name = "Chi-square test of independence"
                
                result = {
                    'type': test_type,
                    'name': test_name,
                    'statistic': chi2_stat,
                    'p_value': p_value,
                    'alpha': alpha,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at alpha={alpha}"
                }
                
            elif test_type == 'correlation':
                # Correlation test
                if data2_values is None:
                    raise ValueError("Correlation test requires two data samples")
                
                corr_coef, p_value = stats.pearsonr(data1_array, data2_array)
                
                result = {
                    'type': test_type,
                    'name': "Pearson correlation test",
                    'statistic': corr_coef,
                    'p_value': p_value,
                    'alpha': alpha,
                    'reject_null': p_value < alpha,
                    'interpretation': f"{'Reject' if p_value < alpha else 'Fail to reject'} the null hypothesis at alpha={alpha}"
                }
                
            else:
                raise ValueError(f"Unsupported test type: {test_type}")
            
            # Store the results
            result_name = f"{dataset or 'custom'}_{test_type}_test"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing hypothesis test: {e}")
            raise
    
    def solve_equation(self, equation: str, variable: str = 'x') -> Dict[str, Any]:
        """
        Solve a symbolic equation.
        
        Args:
            equation: Equation to solve (e.g., "x**2 + 2*x - 3 = 0")
            variable: Variable to solve for
            
        Returns:
            Dict[str, Any]: Solution results
        """
        try:
            # Parse equation
            if "=" in equation:
                left, right = equation.split("=", 1)
                expr = f"({left}) - ({right})"
            else:
                expr = equation
            
            # Get or create symbolic variable
            if variable in self.symbolic_variables:
                var = self.symbolic_variables[variable]
            else:
                var = sp.Symbol(variable)
                self.symbolic_variables[variable] = var
            
            # Parse expression
            sympy_expr = sp.sympify(expr)
            
            # Solve equation
            solutions = sp.solve(sympy_expr, var)
            
            # Convert solutions to strings
            solution_strs = [str(sol) for sol in solutions]
            
            result = {
                'equation': equation,
                'variable': variable,
                'solutions': solution_strs,
                'count': len(solutions)
            }
            
            # Store the results
            result_name = f"equation_solve_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error solving equation: {e}")
            raise
    
    def calculate_expression(self, expression: str, variable_values: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Calculate the value of a symbolic expression.
        
        Args:
            expression: Expression to calculate
            variable_values: Dictionary mapping variable names to values
            
        Returns:
            Dict[str, Any]: Calculation results
        """
        try:
            # Parse expression
            sympy_expr = sp.sympify(expression)
            
            # Substitute variable values if provided
            if variable_values:
                # Create symbols for variables not already in symbolic_variables
                for var_name in variable_values:
                    if var_name not in self.symbolic_variables:
                        self.symbolic_variables[var_name] = sp.Symbol(var_name)
                
                # Create substitution dictionary
                subs_dict = {self.symbolic_variables[var_name]: value 
                            for var_name, value in variable_values.items()
                            if var_name in self.symbolic_variables}
                
                # Substitute values
                result_expr = sympy_expr.subs(subs_dict)
                
                # Try to evaluate to a numerical value
                try:
                    result_value = float(result_expr)
                except:
                    result_value = None
            else:
                result_expr = sympy_expr
                result_value = None
            
            result = {
                'expression': expression,
                'result_expression': str(result_expr),
                'result_value': result_value,
                'variable_values': variable_values
            }
            
            # Store the results
            result_name = f"expression_calc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating expression: {e}")
            raise
    
    def differentiate(self, expression: str, variable: str = 'x', order: int = 1) -> Dict[str, Any]:
        """
        Differentiate a symbolic expression.
        
        Args:
            expression: Expression to differentiate
            variable: Variable to differentiate with respect to
            order: Order of differentiation
            
        Returns:
            Dict[str, Any]: Differentiation results
        """
        try:
            # Get or create symbolic variable
            if variable in self.symbolic_variables:
                var = self.symbolic_variables[variable]
            else:
                var = sp.Symbol(variable)
                self.symbolic_variables[variable] = var
            
            # Parse expression
            sympy_expr = sp.sympify(expression)
            
            # Differentiate
            derivative = sympy_expr
            for _ in range(order):
                derivative = sp.diff(derivative, var)
            
            # Simplify
            derivative = sp.simplify(derivative)
            
            result = {
                'expression': expression,
                'variable': variable,
                'order': order,
                'derivative': str(derivative),
                'derivative_latex': sp.latex(derivative)
            }
            
            # Store the results
            result_name = f"derivative_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error differentiating expression: {e}")
            raise
    
    def integrate(self, expression: str, variable: str = 'x', lower_bound: float = None, upper_bound: float = None) -> Dict[str, Any]:
        """
        Integrate a symbolic expression.
        
        Args:
            expression: Expression to integrate
            variable: Variable to integrate with respect to
            lower_bound: Lower bound for definite integral (optional)
            upper_bound: Upper bound for definite integral (optional)
            
        Returns:
            Dict[str, Any]: Integration results
        """
        try:
            # Get or create symbolic variable
            if variable in self.symbolic_variables:
                var = self.symbolic_variables[variable]
            else:
                var = sp.Symbol(variable)
                self.symbolic_variables[variable] = var
            
            # Parse expression
            sympy_expr = sp.sympify(expression)
            
            # Integrate
            if lower_bound is not None and upper_bound is not None:
                # Definite integral
                integral = sp.integrate(sympy_expr, (var, lower_bound, upper_bound))
                integral_type = 'definite'
            else:
                # Indefinite integral
                integral = sp.integrate(sympy_expr, var)
                integral_type = 'indefinite'
            
            # Simplify
            integral = sp.simplify(integral)
            
            result = {
                'expression': expression,
                'variable': variable,
                'type': integral_type,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'integral': str(integral),
                'integral_latex': sp.latex(integral)
            }
            
            # Store the results
            result_name = f"integral_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error integrating expression: {e}")
            raise
    
    def solve_system(self, equations: List[str], variables: List[str] = None) -> Dict[str, Any]:
        """
        Solve a system of equations.
        
        Args:
            equations: List of equations to solve
            variables: List of variables to solve for (optional)
            
        Returns:
            Dict[str, Any]: Solution results
        """
        try:
            # Parse equations
            sympy_eqs = []
            for eq in equations:
                if "=" in eq:
                    left, right = eq.split("=", 1)
                    sympy_eqs.append(sp.Eq(sp.sympify(left), sp.sympify(right)))
                else:
                    sympy_eqs.append(sp.Eq(sp.sympify(eq), 0))
            
            # Get or create symbolic variables
            if variables is None:
                # Extract variables from equations
                all_symbols = set()
                for eq in sympy_eqs:
                    all_symbols.update(eq.free_symbols)
                vars_list = list(all_symbols)
            else:
                vars_list = []
                for var_name in variables:
                    if var_name in self.symbolic_variables:
                        vars_list.append(self.symbolic_variables[var_name])
                    else:
                        var = sp.Symbol(var_name)
                        self.symbolic_variables[var_name] = var
                        vars_list.append(var)
            
            # Solve system
            solution = sp.solve(sympy_eqs, vars_list)
            
            # Convert solution to dictionary
            if isinstance(solution, list):
                # Multiple solutions
                solution_dicts = []
                for sol in solution:
                    if isinstance(sol, dict):
                        solution_dicts.append({str(var): str(value) for var, value in sol.items()})
                    else:
                        # Handle tuple solutions
                        sol_dict = {}
                        for i, var in enumerate(vars_list):
                            sol_dict[str(var)] = str(sol[i])
                        solution_dicts.append(sol_dict)
                solution_result = solution_dicts
            elif isinstance(solution, dict):
                # Single solution
                solution_result = {str(var): str(value) for var, value in solution.items()}
            else:
                # Empty solution or other format
                solution_result = str(solution)
            
            result = {
                'equations': equations,
                'variables': [str(var) for var in vars_list],
                'solution': solution_result,
                'has_solution': bool(solution)
            }
            
            # Store the results
            result_name = f"system_solve_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error solving system of equations: {e}")
            raise
    
    def run_simulation(self, 
                      model_function: Callable,
                      parameters: Dict[str, float],
                      initial_conditions: Dict[str, float],
                      time_span: Tuple[float, float],
                      time_points: int = 100,
                      name: str = None) -> Dict[str, Any]:
        """
        Run a numerical simulation.
        
        Args:
            model_function: Function defining the model dynamics
            parameters: Dictionary of model parameters
            initial_conditions: Dictionary of initial conditions
            time_span: Tuple of (start_time, end_time)
            time_points: Number of time points to simulate
            name: Name for the simulation
            
        Returns:
            Dict[str, Any]: Simulation results
        """
        try:
            # Generate a name if not provided
            if name is None:
                name = f"simulation_{len(self.simulation_results) + 1}"
            
            # Create time points
            t_start, t_end = time_span
            t = np.linspace(t_start, t_end, time_points)
            
            # Run simulation
            result = model_function(t, initial_conditions, parameters)
            
            # Store results
            simulation_data = {
                'time': t,
                'result': result,
                'parameters': parameters,
                'initial_conditions': initial_conditions,
                'time_span': time_span,
                'time_points': time_points
            }
            
            self.simulation_results[name] = simulation_data
            
            logger.info(f"Simulation '{name}' completed with {time_points} time points")
            return simulation_data
            
        except Exception as e:
            logger.error(f"Error running simulation: {e}")
            raise
    
    def plot_simulation(self, 
                       simulation_name: str,
                       variables: List[str] = None,
                       title: str = None,
                       xlabel: str = 'Time',
                       ylabel: str = 'Value',
                       figsize: Tuple[int, int] = (10, 6),
                       return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot simulation results.
        
        Args:
            simulation_name: Name of the simulation to plot
            variables: List of variables to plot (optional)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            if simulation_name not in self.simulation_results:
                raise ValueError(f"Simulation '{simulation_name}' not found")
            
            simulation = self.simulation_results[simulation_name]
            
            # Get time and result
            t = simulation['time']
            result = simulation['result']
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot variables
            if variables is None:
                # Plot all variables
                if isinstance(result, dict):
                    for var_name, values in result.items():
                        ax.plot(t, values, label=var_name)
                elif isinstance(result, np.ndarray):
                    if len(result.shape) == 1:
                        # Single variable
                        ax.plot(t, result, label='Variable')
                    else:
                        # Multiple variables
                        for i in range(result.shape[1]):
                            ax.plot(t, result[:, i], label=f'Variable {i+1}')
                else:
                    raise ValueError(f"Unsupported result type: {type(result)}")
            else:
                # Plot specified variables
                if isinstance(result, dict):
                    for var_name in variables:
                        if var_name in result:
                            ax.plot(t, result[var_name], label=var_name)
                        else:
                            logger.warning(f"Variable '{var_name}' not found in simulation results")
                elif isinstance(result, np.ndarray):
                    for i, var_name in enumerate(variables):
                        if i < result.shape[1]:
                            ax.plot(t, result[:, i], label=var_name)
                        else:
                            logger.warning(f"Variable index {i} out of range")
                else:
                    raise ValueError(f"Unsupported result type: {type(result)}")
            
            # Set labels and title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Simulation: {simulation_name}")
                
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting simulation: {e}")
            raise
    
    def run_monte_carlo(self, 
                       model_function: Callable,
                       parameter_ranges: Dict[str, Tuple[float, float]],
                       initial_conditions: Dict[str, float],
                       time_span: Tuple[float, float],
                       num_simulations: int = 100,
                       time_points: int = 50,
                       name: str = None) -> Dict[str, Any]:
        """
        Run a Monte Carlo simulation with varying parameters.
        
        Args:
            model_function: Function defining the model dynamics
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            initial_conditions: Dictionary of initial conditions
            time_span: Tuple of (start_time, end_time)
            num_simulations: Number of simulations to run
            time_points: Number of time points per simulation
            name: Name for the Monte Carlo simulation
            
        Returns:
            Dict[str, Any]: Monte Carlo simulation results
        """
        try:
            # Generate a name if not provided
            if name is None:
                name = f"monte_carlo_{len(self.simulation_results) + 1}"
            
            # Create time points
            t_start, t_end = time_span
            t = np.linspace(t_start, t_end, time_points)
            
            # Run simulations
            all_results = []
            all_parameters = []
            
            for i in range(num_simulations):
                # Generate random parameters
                params = {}
                for param_name, (min_val, max_val) in parameter_ranges.items():
                    params[param_name] = np.random.uniform(min_val, max_val)
                
                # Run simulation
                result = model_function(t, initial_conditions, params)
                
                all_results.append(result)
                all_parameters.append(params)
            
            # Store results
            monte_carlo_data = {
                'time': t,
                'results': all_results,
                'parameters': all_parameters,
                'parameter_ranges': parameter_ranges,
                'initial_conditions': initial_conditions,
                'time_span': time_span,
                'num_simulations': num_simulations,
                'time_points': time_points
            }
            
            self.simulation_results[name] = monte_carlo_data
            
            logger.info(f"Monte Carlo simulation '{name}' completed with {num_simulations} simulations")
            return monte_carlo_data
            
        except Exception as e:
            logger.error(f"Error running Monte Carlo simulation: {e}")
            raise
    
    def plot_monte_carlo(self, 
                        simulation_name: str,
                        variable: Union[str, int] = 0,
                        confidence_interval: float = 0.95,
                        title: str = None,
                        xlabel: str = 'Time',
                        ylabel: str = 'Value',
                        figsize: Tuple[int, int] = (10, 6),
                        return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot Monte Carlo simulation results with confidence intervals.
        
        Args:
            simulation_name: Name of the Monte Carlo simulation to plot
            variable: Variable to plot (name or index)
            confidence_interval: Confidence interval (0-1)
            title: Plot title
            xlabel: x-axis label
            ylabel: y-axis label
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            if simulation_name not in self.simulation_results:
                raise ValueError(f"Simulation '{simulation_name}' not found")
            
            simulation = self.simulation_results[simulation_name]
            
            # Check if it's a Monte Carlo simulation
            if 'results' not in simulation or 'parameters' not in simulation:
                raise ValueError(f"'{simulation_name}' is not a Monte Carlo simulation")
            
            # Get time and results
            t = simulation['time']
            all_results = simulation['results']
            
            # Extract the specified variable from all simulations
            variable_results = []
            
            for result in all_results:
                if isinstance(result, dict):
                    # Dictionary result
                    if isinstance(variable, str):
                        if variable in result:
                            variable_results.append(result[variable])
                        else:
                            raise ValueError(f"Variable '{variable}' not found in simulation results")
                    else:
                        # Use first variable if index is provided
                        var_name = list(result.keys())[variable]
                        variable_results.append(result[var_name])
                elif isinstance(result, np.ndarray):
                    # Array result
                    if len(result.shape) == 1:
                        # Single variable
                        variable_results.append(result)
                    else:
                        # Multiple variables
                        if isinstance(variable, int):
                            if variable < result.shape[1]:
                                variable_results.append(result[:, variable])
                            else:
                                raise ValueError(f"Variable index {variable} out of range")
                        else:
                            # Assume first variable if string is provided for array result
                            variable_results.append(result[:, 0])
                else:
                    raise ValueError(f"Unsupported result type: {type(result)}")
            
            # Convert to numpy array
            variable_results = np.array(variable_results)
            
            # Calculate statistics
            mean = np.mean(variable_results, axis=0)
            std = np.std(variable_results, axis=0)
            
            # Calculate confidence interval
            z = stats.norm.ppf(0.5 + confidence_interval / 2)
            margin = z * std / np.sqrt(len(all_results))
            lower = mean - margin
            upper = mean + margin
            
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot individual simulations (transparent)
            for i in range(min(20, len(variable_results))):  # Limit to 20 simulations for clarity
                ax.plot(t, variable_results[i], color='blue', alpha=0.1)
            
            # Plot mean
            ax.plot(t, mean, color='blue', label='Mean')
            
            # Plot confidence interval
            ax.fill_between(t, lower, upper, color='blue', alpha=0.2, label=f'{confidence_interval*100:.0f}% Confidence Interval')
            
            # Set labels and title
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Monte Carlo Simulation: {simulation_name}")
                
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting Monte Carlo simulation: {e}")
            raise
    
    def run_optimization(self, 
                        objective_function: Callable,
                        initial_guess: List[float],
                        bounds: List[Tuple[float, float]] = None,
                        constraints: List[Dict[str, Any]] = None,
                        method: str = 'BFGS') -> Dict[str, Any]:
        """
        Run an optimization to find the minimum of a function.
        
        Args:
            objective_function: Function to minimize
            initial_guess: Initial parameter values
            bounds: List of (min, max) bounds for each parameter
            constraints: List of constraint dictionaries
            method: Optimization method
            
        Returns:
            Dict[str, Any]: Optimization results
        """
        try:
            # Run optimization
            if bounds is not None:
                if method == 'BFGS':
                    # BFGS doesn't support bounds, switch to L-BFGS-B
                    logger.info("Switching to L-BFGS-B method to support bounds")
                    method = 'L-BFGS-B'
                
                result = optimize.minimize(
                    objective_function,
                    initial_guess,
                    method=method,
                    bounds=bounds,
                    constraints=constraints
                )
            else:
                result = optimize.minimize(
                    objective_function,
                    initial_guess,
                    method=method,
                    constraints=constraints
                )
            
            # Extract results
            optimization_result = {
                'success': result.success,
                'message': result.message,
                'x': result.x.tolist(),
                'fun': float(result.fun),
                'nit': int(result.nit) if hasattr(result, 'nit') else None,
                'nfev': int(result.nfev),
                'method': method
            }
            
            # Store the results
            result_name = f"optimization_{datetime.now().strftime('%Y%m%d%H%M%S')}"
            self.analysis_results[result_name] = optimization_result
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error running optimization: {e}")
            raise
    
    def run_pca(self, 
               data: Union[str, pd.DataFrame],
               n_components: int = None,
               standardize: bool = True) -> Dict[str, Any]:
        """
        Perform Principal Component Analysis (PCA).
        
        Args:
            data: Dataset name or DataFrame
            n_components: Number of components to keep (default: all)
            standardize: Whether to standardize the data
            
        Returns:
            Dict[str, Any]: PCA results
        """
        try:
            # Get data
            if isinstance(data, str):
                if data not in self.datasets:
                    raise ValueError(f"Dataset '{data}' not found")
                df = self.datasets[data]
            else:
                df = data
            
            # Remove non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] == 0:
                raise ValueError("No numeric columns found in dataset")
            
            # Standardize data if requested
            if standardize:
                X = (numeric_df - numeric_df.mean()) / numeric_df.std()
            else:
                X = numeric_df
            
            # Fill NaN values with 0
            X = X.fillna(0)
            
            # Perform PCA
            from sklearn.decomposition import PCA
            
            if n_components is None:
                n_components = min(X.shape)
            
            pca = PCA(n_components=n_components)
            principal_components = pca.fit_transform(X)
            
            # Create DataFrame with principal components
            pc_df = pd.DataFrame(
                data=principal_components,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            
            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance)
            
            # Get component loadings
            loadings = pca.components_
            loadings_df = pd.DataFrame(
                data=loadings.T,
                columns=[f'PC{i+1}' for i in range(n_components)],
                index=numeric_df.columns
            )
            
            # Create result
            result = {
                'n_components': n_components,
                'explained_variance': explained_variance.tolist(),
                'cumulative_variance': cumulative_variance.tolist(),
                'loadings': loadings_df.to_dict(),
                'principal_components': pc_df.to_dict(),
                'standardized': standardize
            }
            
            # Store the results
            result_name = f"{data if isinstance(data, str) else 'custom'}_pca"
            self.analysis_results[result_name] = result
            
            # Store the principal components as a new dataset
            pc_dataset_name = f"{data if isinstance(data, str) else 'custom'}_pca_components"
            self.datasets[pc_dataset_name] = pc_df
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing PCA: {e}")
            raise
    
    def plot_pca(self, 
                pca_result: Dict[str, Any],
                plot_type: str = 'variance',
                components: List[int] = [0, 1],
                figsize: Tuple[int, int] = (10, 6),
                return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot PCA results.
        
        Args:
            pca_result: PCA result dictionary
            plot_type: Type of plot ('variance', 'components', 'loadings', 'biplot')
            components: List of component indices to plot (for 'components' and 'biplot')
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Create figure
            fig, ax = plt.subplots(figsize=figsize)
            
            if plot_type == 'variance':
                # Plot explained variance
                explained_variance = pca_result['explained_variance']
                cumulative_variance = pca_result['cumulative_variance']
                
                x = range(1, len(explained_variance) + 1)
                
                ax.bar(x, explained_variance, alpha=0.7, label='Explained Variance')
                ax.step(x, cumulative_variance, where='mid', color='red', label='Cumulative Variance')
                
                ax.set_xlabel('Principal Component')
                ax.set_ylabel('Explained Variance Ratio')
                ax.set_title('PCA Explained Variance')
                ax.set_xticks(x)
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                
            elif plot_type == 'components':
                # Plot principal components
                pc_df = pd.DataFrame(pca_result['principal_components'])
                
                if len(components) < 2:
                    raise ValueError("Need at least 2 components to plot")
                
                pc1, pc2 = components[:2]
                pc1_name = f'PC{pc1+1}'
                pc2_name = f'PC{pc2+1}'
                
                if pc1_name not in pc_df.columns or pc2_name not in pc_df.columns:
                    raise ValueError(f"Components {pc1_name} or {pc2_name} not found in PCA results")
                
                ax.scatter(pc_df[pc1_name], pc_df[pc2_name], alpha=0.7)
                
                ax.set_xlabel(f'{pc1_name} ({pca_result["explained_variance"][pc1]:.2%})')
                ax.set_ylabel(f'{pc2_name} ({pca_result["explained_variance"][pc2]:.2%})')
                ax.set_title('PCA Components')
                ax.grid(True, linestyle='--', alpha=0.7)
                
            elif plot_type == 'loadings':
                # Plot component loadings
                loadings_df = pd.DataFrame(pca_result['loadings'])
                
                if len(components) < 1:
                    raise ValueError("Need at least 1 component to plot")
                
                pc = components[0]
                pc_name = f'PC{pc+1}'
                
                if pc_name not in loadings_df.columns:
                    raise ValueError(f"Component {pc_name} not found in PCA results")
                
                # Sort loadings
                sorted_loadings = loadings_df[pc_name].sort_values()
                
                # Plot horizontal bar chart
                ax.barh(sorted_loadings.index, sorted_loadings, alpha=0.7)
                
                ax.set_xlabel(f'Loading on {pc_name}')
                ax.set_title(f'PCA Loadings for {pc_name}')
                ax.grid(True, linestyle='--', alpha=0.7)
                
            elif plot_type == 'biplot':
                # Create biplot of components and loadings
                pc_df = pd.DataFrame(pca_result['principal_components'])
                loadings_df = pd.DataFrame(pca_result['loadings'])
                
                if len(components) < 2:
                    raise ValueError("Need at least 2 components to plot")
                
                pc1, pc2 = components[:2]
                pc1_name = f'PC{pc1+1}'
                pc2_name = f'PC{pc2+1}'
                
                if pc1_name not in pc_df.columns or pc2_name not in pc_df.columns:
                    raise ValueError(f"Components {pc1_name} or {pc2_name} not found in PCA results")
                
                # Plot principal components
                ax.scatter(pc_df[pc1_name], pc_df[pc2_name], alpha=0.7)
                
                # Plot loadings as vectors
                for i, feature in enumerate(loadings_df.index):
                    ax.arrow(0, 0, 
                            loadings_df.loc[feature, pc1_name] * 5,  # Scale for visibility
                            loadings_df.loc[feature, pc2_name] * 5,
                            head_width=0.1, head_length=0.1, fc='red', ec='red')
                    ax.text(loadings_df.loc[feature, pc1_name] * 5.2,
                           loadings_df.loc[feature, pc2_name] * 5.2,
                           feature)
                
                ax.set_xlabel(f'{pc1_name} ({pca_result["explained_variance"][pc1]:.2%})')
                ax.set_ylabel(f'{pc2_name} ({pca_result["explained_variance"][pc2]:.2%})')
                ax.set_title('PCA Biplot')
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Set equal aspect ratio
                ax.set_aspect('equal')
                
                # Add circle
                circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
                ax.add_patch(circle)
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting PCA results: {e}")
            raise
    
    def run_cluster_analysis(self, 
                            data: Union[str, pd.DataFrame],
                            n_clusters: int = 3,
                            method: str = 'kmeans',
                            standardize: bool = True) -> Dict[str, Any]:
        """
        Perform cluster analysis.
        
        Args:
            data: Dataset name or DataFrame
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')
            standardize: Whether to standardize the data
            
        Returns:
            Dict[str, Any]: Clustering results
        """
        try:
            # Get data
            if isinstance(data, str):
                if data not in self.datasets:
                    raise ValueError(f"Dataset '{data}' not found")
                df = self.datasets[data]
            else:
                df = data
            
            # Remove non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] == 0:
                raise ValueError("No numeric columns found in dataset")
            
            # Standardize data if requested
            if standardize:
                X = (numeric_df - numeric_df.mean()) / numeric_df.std()
            else:
                X = numeric_df
            
            # Fill NaN values with 0
            X = X.fillna(0)
            
            # Perform clustering
            if method == 'kmeans':
                from sklearn.cluster import KMeans
                
                model = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = model.fit_predict(X)
                
                # Get cluster centers
                centers = model.cluster_centers_
                
                # Calculate inertia (within-cluster sum of squares)
                inertia = model.inertia_
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X, clusters)
                
                method_name = "K-Means"
                
            elif method == 'hierarchical':
                from sklearn.cluster import AgglomerativeClustering
                
                model = AgglomerativeClustering(n_clusters=n_clusters)
                clusters = model.fit_predict(X)
                
                # No cluster centers for hierarchical clustering
                centers = None
                
                # No inertia for hierarchical clustering
                inertia = None
                
                # Calculate silhouette score
                from sklearn.metrics import silhouette_score
                silhouette = silhouette_score(X, clusters)
                
                method_name = "Hierarchical Clustering"
                
            elif method == 'dbscan':
                from sklearn.cluster import DBSCAN
                
                # DBSCAN doesn't use n_clusters directly
                model = DBSCAN(eps=0.5, min_samples=5)
                clusters = model.fit_predict(X)
                
                # No cluster centers for DBSCAN
                centers = None
                
                # No inertia for DBSCAN
                inertia = None
                
                # Calculate silhouette score if more than one cluster
                if len(np.unique(clusters)) > 1 and -1 not in np.unique(clusters):
                    from sklearn.metrics import silhouette_score
                    silhouette = silhouette_score(X, clusters)
                else:
                    silhouette = None
                
                # Update n_clusters to actual number of clusters found
                n_clusters = len(np.unique(clusters))
                if -1 in clusters:  # DBSCAN marks outliers as -1
                    n_clusters -= 1
                
                method_name = "DBSCAN"
                
            else:
                raise ValueError(f"Unsupported clustering method: {method}")
            
            # Add cluster labels to original dataframe
            cluster_df = df.copy()
            cluster_df['cluster'] = clusters
            
            # Create result
            result = {
                'method': method_name,
                'n_clusters': n_clusters,
                'clusters': clusters.tolist(),
                'centers': centers.tolist() if centers is not None else None,
                'inertia': float(inertia) if inertia is not None else None,
                'silhouette': float(silhouette) if silhouette is not None else None,
                'standardized': standardize
            }
            
            # Store the results
            result_name = f"{data if isinstance(data, str) else 'custom'}_{method}_clustering"
            self.analysis_results[result_name] = result
            
            # Store the clustered dataset
            cluster_dataset_name = f"{data if isinstance(data, str) else 'custom'}_clustered"
            self.datasets[cluster_dataset_name] = cluster_df
            
            return result
            
        except Exception as e:
            logger.error(f"Error performing cluster analysis: {e}")
            raise
    
    def plot_clusters(self, 
                     data: Union[str, pd.DataFrame],
                     cluster_result: Dict[str, Any],
                     features: List[str] = None,
                     plot_type: str = 'scatter',
                     figsize: Tuple[int, int] = (10, 6),
                     return_base64: bool = True) -> Union[str, plt.Figure]:
        """
        Plot clustering results.
        
        Args:
            data: Dataset name or DataFrame
            cluster_result: Clustering result dictionary
            features: List of features to plot (default: first two)
            plot_type: Type of plot ('scatter', 'pca', '3d')
            figsize: Figure size as (width, height)
            return_base64: If True, return base64-encoded image; otherwise return Figure
            
        Returns:
            Union[str, plt.Figure]: Base64-encoded image or matplotlib Figure
        """
        try:
            # Get data
            if isinstance(data, str):
                if data not in self.datasets:
                    raise ValueError(f"Dataset '{data}' not found")
                df = self.datasets[data]
            else:
                df = data
            
            # Get clusters
            clusters = np.array(cluster_result['clusters'])
            
            # Get numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            if numeric_df.shape[1] == 0:
                raise ValueError("No numeric columns found in dataset")
            
            # Select features to plot
            if features is None:
                # Use first two numeric columns
                features = numeric_df.columns[:2].tolist()
            
            # Check if features exist
            for feature in features:
                if feature not in df.columns:
                    raise ValueError(f"Feature '{feature}' not found in dataset")
            
            # Create figure
            if plot_type == '3d' and len(features) >= 3:
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(111, projection='3d')
            else:
                fig, ax = plt.subplots(figsize=figsize)
            
            # Plot based on type
            if plot_type == 'scatter':
                # 2D scatter plot
                if len(features) < 2:
                    raise ValueError("Need at least 2 features for scatter plot")
                
                # Get unique clusters
                unique_clusters = np.unique(clusters)
                
                # Plot each cluster
                for cluster_id in unique_clusters:
                    mask = clusters == cluster_id
                    label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Outliers'
                    color = 'gray' if cluster_id == -1 else None
                    
                    ax.scatter(
                        df.loc[mask, features[0]],
                        df.loc[mask, features[1]],
                        alpha=0.7,
                        label=label,
                        color=color
                    )
                
                # Plot cluster centers if available
                centers = cluster_result.get('centers')
                if centers is not None:
                    centers = np.array(centers)
                    
                    # Find indices of the selected features
                    feature_indices = [list(numeric_df.columns).index(feature) for feature in features[:2]]
                    
                    # Plot centers
                    ax.scatter(
                        centers[:, feature_indices[0]],
                        centers[:, feature_indices[1]],
                        s=100,
                        marker='X',
                        color='black',
                        label='Cluster Centers'
                    )
                
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                
            elif plot_type == 'pca':
                # PCA scatter plot
                from sklearn.decomposition import PCA
                
                # Get data for PCA
                X = df[features].values
                
                # Standardize
                X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                
                # Perform PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                
                # Get unique clusters
                unique_clusters = np.unique(clusters)
                
                # Plot each cluster
                for cluster_id in unique_clusters:
                    mask = clusters == cluster_id
                    label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Outliers'
                    color = 'gray' if cluster_id == -1 else None
                    
                    ax.scatter(
                        X_pca[mask, 0],
                        X_pca[mask, 1],
                        alpha=0.7,
                        label=label,
                        color=color
                    )
                
                # Plot cluster centers if available
                centers = cluster_result.get('centers')
                if centers is not None:
                    # Project centers to PCA space
                    centers_pca = pca.transform(centers)
                    
                    # Plot centers
                    ax.scatter(
                        centers_pca[:, 0],
                        centers_pca[:, 1],
                        s=100,
                        marker='X',
                        color='black',
                        label='Cluster Centers'
                    )
                
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
                
            elif plot_type == '3d':
                # 3D scatter plot
                if len(features) < 3:
                    raise ValueError("Need at least 3 features for 3D plot")
                
                # Get unique clusters
                unique_clusters = np.unique(clusters)
                
                # Plot each cluster
                for cluster_id in unique_clusters:
                    mask = clusters == cluster_id
                    label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Outliers'
                    color = 'gray' if cluster_id == -1 else None
                    
                    ax.scatter(
                        df.loc[mask, features[0]],
                        df.loc[mask, features[1]],
                        df.loc[mask, features[2]],
                        alpha=0.7,
                        label=label,
                        color=color
                    )
                
                # Plot cluster centers if available
                centers = cluster_result.get('centers')
                if centers is not None:
                    centers = np.array(centers)
                    
                    # Find indices of the selected features
                    feature_indices = [list(numeric_df.columns).index(feature) for feature in features[:3]]
                    
                    # Plot centers
                    ax.scatter(
                        centers[:, feature_indices[0]],
                        centers[:, feature_indices[1]],
                        centers[:, feature_indices[2]],
                        s=100,
                        marker='X',
                        color='black',
                        label='Cluster Centers'
                    )
                
                ax.set_xlabel(features[0])
                ax.set_ylabel(features[1])
                ax.set_zlabel(features[2])
                
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            # Set title
            method_name = cluster_result.get('method', 'Clustering')
            n_clusters = cluster_result.get('n_clusters', 'unknown')
            
            if plot_type == 'pca':
                ax.set_title(f'{method_name} (PCA projection) with {n_clusters} clusters')
            else:
                ax.set_title(f'{method_name} with {n_clusters} clusters')
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            # Tight layout
            fig.tight_layout()
            
            # Increment figure counter
            self.figure_counter += 1
            
            if return_base64:
                # Convert to base64
                buf = io.BytesIO()
                fig.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                plt.close(fig)
                return img_str
            else:
                return fig
                
        except Exception as e:
            logger.error(f"Error plotting clusters: {e}")
            raise
    
    def run(self, command: str = None, **kwargs) -> Any:
        """
        Run a command or return help information.
        
        Args:
            command: Command to run (optional)
            **kwargs: Additional arguments
            
        Returns:
            Any: Command result or help information
        """
        if command is None:
            # Return help information
            return self.help()
        
        # Parse and execute command
        try:
            # Check if command is a method name
            if hasattr(self, command) and callable(getattr(self, command)):
                method = getattr(self, command)
                return method(**kwargs)
            
            # Otherwise, try to evaluate as a Python expression
            # This is potentially dangerous and should be used with caution
            return eval(command)
            
        except Exception as e:
            logger.error(f"Error executing command '{command}': {e}")
            return f"Error: {str(e)}"
    
    def help(self) -> str:
        """Return help information about the ONI Science Lab."""
        help_text = """
        ONI Science Lab - Scientific computing and analysis tools
        
        Available methods:
        - load_dataset(data, name): Load a dataset
        - get_dataset(name): Get a dataset by name
        - list_datasets(): List all available datasets
        - describe_dataset(name): Get descriptive statistics for a dataset
        - plot_data(x, y, dataset, plot_type, ...): Create a plot of data
        - perform_regression(x, y, dataset, reg_type, ...): Perform regression analysis
        - perform_hypothesis_test(data1, data2, dataset, test_type, ...): Perform a hypothesis test
        - solve_equation(equation, variable): Solve a symbolic equation
        - calculate_expression(expression, variable_values): Calculate the value of a symbolic expression
        - differentiate(expression, variable, order): Differentiate a symbolic expression
        - integrate(expression, variable, lower_bound, upper_bound): Integrate a symbolic expression
        - solve_system(equations, variables): Solve a system of equations
        - run_simulation(model_function, parameters, initial_conditions, ...): Run a numerical simulation
        - plot_simulation(simulation_name, variables, ...): Plot simulation results
        - run_monte_carlo(model_function, parameter_ranges, ...): Run a Monte Carlo simulation
        - plot_monte_carlo(simulation_name, variable, ...): Plot Monte Carlo simulation results
        - run_optimization(objective_function, initial_guess, ...): Run an optimization
        - run_pca(data, n_components, standardize): Perform Principal Component Analysis
        - plot_pca(pca_result, plot_type, ...): Plot PCA results
        - run_cluster_analysis(data, n_clusters, method, ...): Perform cluster analysis
        - plot_clusters(data, cluster_result, features, ...): Plot clustering results
        
        For more information on a specific method, use help(ONIScienceLab.method_name)
        """
        return help_text
    
    def cleanup(self):
        """Clean up resources."""
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear datasets and results
        self.datasets.clear()
        self.analysis_results.clear()
        self.simulation_results.clear()
        
        logger.info("ONI Science Lab cleaned up")

# Example usage
if __name__ == "__main__":
    lab = ONIScienceLab()
    
    # Create a sample dataset
    data = {
        'x': np.linspace(0, 10, 100),
        'y': np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    }
    
    df = lab.load_dataset(data, 'sine_wave')
    
    # Perform regression
    result = lab.perform_regression('x', 'y', 'sine_wave', 'polynomial', degree=3)
    
    # Plot the data and regression
    plot = lab.plot_data('x', 'y', 'sine_wave', 'scatter', title='Sine Wave with Noise')
    
    # Solve an equation
    solution = lab.solve_equation('x**2 - 4*x + 4 = 0', 'x')
    
    print(f"Regression result: {result['equation']}")
    print(f"Equation solution: {solution['solutions']}")