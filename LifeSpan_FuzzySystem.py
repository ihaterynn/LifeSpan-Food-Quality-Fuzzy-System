import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

def initialize_system():
    temperature = ctrl.Antecedent(np.arange(0, 31, 0.1), 'temperature')
    humidity = ctrl.Antecedent(np.arange(0, 101, 0.1), 'humidity')
    food_type = ctrl.Antecedent(np.arange(0, 2, 1), 'food_type')
    time_on_shelf = ctrl.Antecedent(np.arange(0, 31, 0.1), 'time_on_shelf')
    quality = ctrl.Consequent(np.arange(0, 100, 0.1), 'quality')

    temperature['cold'] = mf.trimf(temperature.universe, [0, 0, 15])
    temperature['optimal'] = mf.trimf(temperature.universe, [10, 20, 30])
    temperature['warm'] = mf.trimf(temperature.universe, [25, 30, 30])

    humidity['low'] = mf.trimf(humidity.universe, [0, 0, 40])
    humidity['moderate'] = mf.trimf(humidity.universe, [30, 50, 70])
    humidity['high'] = mf.trimf(humidity.universe, [60, 100, 100])

    food_type['dry'] = mf.trimf(food_type.universe, [0, 0, 1])
    food_type['fresh'] = mf.trimf(food_type.universe, [0, 1, 1])

    time_on_shelf['fresh'] = mf.trimf(time_on_shelf.universe, [0, 0, 3])
    time_on_shelf['acceptable'] = mf.trimf(time_on_shelf.universe, [2, 10, 15])
    time_on_shelf['stale'] = mf.trimf(time_on_shelf.universe, [12, 30, 30])

    quality['poor'] = mf.trimf(quality.universe, [0, 0, 30])
    quality['fair'] = mf.trimf(quality.universe, [25, 50, 75])
    quality['good'] = mf.trimf(quality.universe, [70, 85, 95])
    quality['excellent'] = mf.trimf(quality.universe, [90, 100, 100])

    return temperature, humidity, food_type, time_on_shelf, quality

def define_rules(temperature, humidity, food_type, time_on_shelf, quality):
    rules = []

    # Rules for Dry/Canned Food - more lenient because they are less perishable
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['dry'] & time_on_shelf['fresh'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['dry'] & time_on_shelf['acceptable'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['dry'] & time_on_shelf['stale'], quality['good']))
    
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['fresh'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['acceptable'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['stale'], quality['good']))
    
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['dry'] & time_on_shelf['fresh'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['dry'] & time_on_shelf['acceptable'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['dry'] & time_on_shelf['stale'], quality['good']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['dry'] & time_on_shelf['fresh'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['dry'] & time_on_shelf['acceptable'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['dry'] & time_on_shelf['stale'], quality['good']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['acceptable'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['stale'], quality['fair']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['dry'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['dry'] & time_on_shelf['acceptable'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['dry'] & time_on_shelf['stale'], quality['fair']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['dry'] & time_on_shelf['fresh'], quality['excellent']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['dry'] & time_on_shelf['acceptable'], quality['good']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['dry'] & time_on_shelf['stale'], quality['fair']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['dry'] & time_on_shelf['stale'], quality['fair']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['dry'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['dry'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['dry'] & time_on_shelf['stale'], quality['poor']))

    # Rules for Fresh Food - stricter because fresh food is more perishable
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['fresh'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['low'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['fresh'] & time_on_shelf['fresh'], quality['fair']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['poor']))
    rules.append(ctrl.Rule(temperature['cold'] & humidity['high'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['fresh'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['low'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['fresh'], quality['good']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['fresh'] & time_on_shelf['fresh'], quality['fair']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['poor']))
    rules.append(ctrl.Rule(temperature['optimal'] & humidity['high'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['fresh'] & time_on_shelf['fresh'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['low'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['fresh'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['poor']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['moderate'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))
    
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['fresh'] & time_on_shelf['fresh'], quality['fair']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['fresh'] & time_on_shelf['acceptable'], quality['poor']))
    rules.append(ctrl.Rule(temperature['warm'] & humidity['high'] & food_type['fresh'] & time_on_shelf['stale'], quality['poor']))

    return rules


def construct_fuzzy_control_system(temperature, humidity, food_type, time_on_shelf, quality, rules):
    food_ctrl = ctrl.ControlSystem(rules=rules)
    food = ctrl.ControlSystemSimulation(control_system=food_ctrl)
    return food

def get_user_input():
    while True:
        try:
            temperature_value = float(input("Please enter the temperature (0-30 °C): "))
            humidity_value = float(input("Please enter the humidity (0-100%): "))
            food_type_value = int(input("Enter food type (0 for Dry/Canned, 1 for Fresh): "))
            time_on_shelf_value = float(input("Please enter the time on shelf (0-30 days): "))
            
            if (0 <= temperature_value <= 30 and 
                0 <= humidity_value <= 100 and 
                0 <= food_type_value <= 1 and 
                0 <= time_on_shelf_value <= 30):
                return temperature_value, humidity_value, food_type_value, time_on_shelf_value
            else:
                print("Please enter values within the specified limits!\n")
        except ValueError:
            print("Please enter valid numeric values!\n")

def compute_food_quality(temperature_value, humidity_value, food_type_value, time_on_shelf_value, food):
    food.input['temperature'] = temperature_value
    food.input['humidity'] = humidity_value
    food.input['food_type'] = food_type_value
    food.input['time_on_shelf'] = time_on_shelf_value
    food.compute()
    
    quality_crisp = food.output['quality']
    print(f"Food Quality: {quality_crisp:.2f}%")
    if quality_crisp > 70:
        print("Food is Fresh!")
    elif quality_crisp > 30:
        print("Food is Acceptable!")
    else:
        print("Food is Spoiled!")
    
    quality.view(sim=food)
    plt.show()

def plot_membership_functions(temperature, humidity, food_type, time_on_shelf, quality):
    print("Showing Fuzzy Sets..")
    temperature.view()
    humidity.view()
    food_type.view()
    time_on_shelf.view()
    quality.view()
    plt.show()

def plot_3d_surface(temperature, humidity, food):
    print("Showing 3D Surface Plot..")
    x, y = np.meshgrid(np.linspace(temperature.universe.min(), temperature.universe.max(), 100),
                       np.linspace(humidity.universe.min(), humidity.universe.max(), 100))
    z_quality = np.zeros_like(x, dtype=float)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            food.input['temperature'] = x[i, j]
            food.input['humidity'] = y[i, j]
            food.input['food_type'] = 0  
            food.input['time_on_shelf'] = 10
            food.compute()
            z_quality[i, j] = food.output['quality']
            
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, z_quality, rstride=1, cstride=1, cmap='viridis', linewidth=0.4, antialiased=True)
    
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Humidity (%)')
    ax.set_zlabel('Food Quality (%)')
    ax.set_title('3D Surface Plot of Food Quality')
    plt.show()

def welcome_message():
    print("="*72)
    print("\033[1;34m" + " " * 12 + "✨ Welcome to the Fuzzy Food Shelf Life Quality Monitoring System ✨" + "\033[0m")
    print("="*72)
    print("\nThis system evaluates food quality based on temperature, humidity, food type, and time on shelf.")
    print("Please provide the necessary information to assess the quality of the stored food.\n")

def display_results(quality_crisp):
    # Displaying results with a delay for better user experience
    print("\nCalculating food quality...")
    time.sleep(1)
    print("\nEvaluating storage conditions...")
    time.sleep(1)

    # Display the final quality with a visual effect
    print("\n\033[1;32m" + "✔️  Food Quality: {:.2f}%".format(quality_crisp) + "\033[0m")
    if quality_crisp > 70:
        print("\033[1;32m" + "🟢 The food is Fresh!" + "\033[0m")
    elif quality_crisp > 30:
        print("\033[1;33m" + "🟡 The food is Acceptable." + "\033[0m")
    else:
        print("\033[1;31m" + "🔴 The food is Spoiled!" + "\033[0m")
    
    print("\n" + "="*72 + "\n")

def goodbye_message():
    print("\033[1;34m" + "\n🌟 Thank you for using the Fuzzy Food Shelf Life Quality Monitoring System! 🌟" + "\033[0m")
    print("\033[1;34m" + "="*72 + "\033[0m")
    print("Goodbye and stay safe!\n")

# Main loop with improved interface
if __name__ == "__main__":
    temperature, humidity, food_type, time_on_shelf, quality = initialize_system()
    rules = define_rules(temperature, humidity, food_type, time_on_shelf, quality)
    food = construct_fuzzy_control_system(temperature, humidity, food_type, time_on_shelf, quality, rules)
    
    welcome_message()
    
    while True:
        temperature_value, humidity_value, food_type_value, time_on_shelf_value = get_user_input()
        compute_food_quality(temperature_value, humidity_value, food_type_value, time_on_shelf_value, food)
        
        check_again = input("\nWould you like to check another food item? (yes/no): ").strip().lower()
        if check_again != 'yes':
            break
    
    goodbye_message()
    plot_membership_functions(temperature, humidity, food_type, time_on_shelf, quality)
    plot_3d_surface(temperature, humidity, food)
