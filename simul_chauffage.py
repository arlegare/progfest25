import numpy as np
from scipy.integrate import odeint

def thermal_model(T, t, C, K, K_ext, T_ext_func, P_func):
    """
    Differential equation for the thermal model.
    
    Parameters:
    - T: Temperature vector
    - t: Time
    - C: Heat capacity matrix
    - K: Thermal conductance matrix
    - K_ext: External thermal conductance matrix
    - T_ext_func: Function for external temperature vector
    - P_func: Function for heating power vector
    
    Returns:
    - dTdt: Time derivative of temperature vector
    """
    T_ext = T_ext_func(t)
    P = P_func(t, T)
    dTdt = np.linalg.inv(C) @ (P - K @ T + K_ext @ (T_ext - T))
    return dTdt

def simulate_thermal_model(C, K, K_ext, T_ext_func, P_func, T0, t):
    """
    Simulate the thermal model.
    
    Parameters:
    - C: Heat capacity matrix
    - K: Thermal conductance matrix
    - K_ext: External thermal conductance matrix
    - T_ext_func: Function for external temperature vector
    - P_func: Function for heating power vector
    - T0: Initial temperature vector
    - t: Time vector
    
    Returns:
    - T: Temperature matrix over time
    """
    # Define the function to pass to odeint
    def model(T, t):
        return thermal_model(T, t, C, K, K_ext, T_ext_func, P_func)
    
    # Solve the differential equation
    T = odeint(model, T0, t)
    
    return T

"""
# Example usage:
if __name__ == "__main__":
    # Define parameters
    n_rooms = 3  # Number of rooms
    C = np.diag([1.0, 1.0, 1.0])  # Heat capacity matrix (example values)
    K = np.array([[2.0, -1.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, -1.0, 2.0]])  # Thermal conductance matrix (example values)
    K_ext = np.diag([0.5, 0.5, 0.5])  # External thermal conductance matrix (example values)
    
    # Define external temperature function (example: constant external temperature)
    def T_ext_func(t):
        return np.array([10.0, 10.0, 10.0])
    
    # Define heating power function (example: proportional control)
    def P_func(t, T):
        T_desired = np.array([20.0, 20.0, 20.0])
        K_p = 10.0
        return K_p * (T_desired - T)
    
    # Initial temperature vector (example values)
    T0 = np.array([15.0, 15.0, 15.0])
    
    # Time vector (example values)
    t = np.linspace(0, 10, 100)
    
    # Simulate the model
    T = simulate_thermal_model(C, K, K_ext, T_ext_func, P_func, T0, t)
"""