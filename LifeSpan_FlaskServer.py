from flask import Flask, render_template, request
import skfuzzy as fuzz
import numpy as np
from skfuzzy import control as ctrl
from skfuzzy import membership as mf
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg' to avoid issues with non-main threads
import matplotlib.pyplot as plt
import io
import base64
from mpl_toolkits.mplot3d import Axes3D


app = Flask(__name__)

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

def compute_food_quality(temperature_value, humidity_value, food_type_value, time_on_shelf_value):
    # Reinitialize the system and rules for every request
    temperature, humidity, food_type, time_on_shelf, quality = initialize_system()
    rules = define_rules(temperature, humidity, food_type, time_on_shelf, quality)
    food = construct_fuzzy_control_system(temperature, humidity, food_type, time_on_shelf, quality, rules)

    # Set inputs and compute
    food.input['temperature'] = temperature_value
    food.input['humidity'] = humidity_value
    food.input['food_type'] = food_type_value
    food.input['time_on_shelf'] = time_on_shelf_value
    food.compute()
    return food.output['quality']

def plot_membership_functions(temperature, humidity, food_type, time_on_shelf, quality, quality_value):
    # Clear any previous plots to avoid issues
    plt.clf()
    plt.close('all')

    # Set a smaller figure size for a more compact visualization
    plt.figure(figsize=(8, 5))

    # Custom colors for membership functions
    colors = {'poor': 'red', 'fair': 'orange', 'good': 'blue', 'excellent': 'green'}

    # Plot each term manually with the corresponding color
    for term, membership in quality.terms.items():
        plt.plot(quality.universe, membership.mf, label=term, color=colors.get(term, 'black'), linewidth=2, zorder=1)

    # Shade the activated area under the membership function
    for term, membership in quality.terms.items():
        # Get the activation value using skfuzzy's interp_membership
        activation = fuzz.interp_membership(quality.universe, membership.mf, quality_value)
        # Shade the area up to the activation level using the corresponding color
        plt.fill_between(quality.universe, 0, np.minimum(membership.mf, activation), alpha=0.5, color=colors.get(term, 'orange'), zorder=2)

    # Draw a vertical line to indicate the defuzzified output
    plt.axvline(x=quality_value, color='r', linestyle='--', linewidth=2, zorder=3)
    plt.title(f"Quality Output - {quality_value:.2f}%")
    plt.legend(loc='best')

    # Set axis limits to make sure the graph is centered and occupies the available space
    plt.xlim(0, 100)
    plt.ylim(0, 1.1)

    # Apply tight layout for better spacing
    plt.tight_layout()

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode('utf8')

def plot_individual_membership_functions(temperature, humidity, food_type, time_on_shelf):
    # Dictionary to hold base64 images for each membership function plot
    images = {}
    
    # Set the figure size to a larger size
    figure_size = (10, 6)  # Increased size for better visibility

    # Plot and store membership functions for each variable
    for var_name, variable in zip(
        ["temperature", "humidity", "food_type", "time_on_shelf"],
        [temperature, humidity, food_type, time_on_shelf]
    ):
        plt.figure(figsize=figure_size)  # Set the figure size
        for term, mf in variable.terms.items():
            plt.plot(variable.universe, mf.mf, label=term)
        plt.title(f"{var_name.capitalize()} Membership Functions")
        plt.xlabel(var_name.capitalize())
        plt.ylabel("Membership")
        plt.legend(loc="upper right")
        plt.tight_layout()

        # Save the figure to a bytes buffer and encode to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        images[var_name] = base64.b64encode(img.getvalue()).decode('utf8')
        plt.close()

    return images

def plot_3d_surface(temperature, humidity, food_type, time_on_shelf):
    print("Starting 3D surface plot computations...")

    # First plot for Temperature vs. Humidity vs. Quality
    temp_values = np.linspace(0, 30, 5)
    hum_values = np.linspace(0, 100, 5)
    quality_values_temp_hum = np.zeros((len(temp_values), len(hum_values)))

    for i, temp in enumerate(temp_values):
        for j, hum in enumerate(hum_values):
            quality_values_temp_hum[i, j] = compute_food_quality(temp, hum, 0, 0)  # Use default values for food type and time on shelf

    fig = plt.figure(figsize=(15, 7))

    # First subplot
    ax1 = fig.add_subplot(121, projection='3d')
    X_temp, Y_hum = np.meshgrid(temp_values, hum_values)
    surf1 = ax1.plot_surface(X_temp, Y_hum, quality_values_temp_hum.T, cmap='viridis', edgecolor='none')
    ax1.set_title('3D Surface Plot of Food Quality (Temp vs Humidity)')
    ax1.set_xlabel('Temperature (°C)')
    ax1.set_ylabel('Humidity (%)')
    ax1.set_zlabel('Food Quality (%)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)

    # Second plot for Food Type vs. Time on Shelf vs. Quality
    type_values = np.array([0, 1])  # Food type values: 0 = Dry, 1 = Fresh
    shelf_values = np.linspace(0, 30, 5)
    quality_values_type_shelf = np.zeros((len(type_values), len(shelf_values)))

    for i, f_type in enumerate(type_values):
        for j, shelf in enumerate(shelf_values):
            quality_values_type_shelf[i, j] = compute_food_quality(15, 50, f_type, shelf)  # Use typical values for temperature and humidity

    # Second subplot
    ax2 = fig.add_subplot(122, projection='3d')
    X_type, Y_shelf = np.meshgrid(type_values, shelf_values)
    surf2 = ax2.plot_surface(X_type, Y_shelf, quality_values_type_shelf.T, cmap='plasma', edgecolor='none')
    ax2.set_title('3D Surface Plot of Food Quality (Food Type vs Time on Shelf)')
    ax2.set_xlabel('Food Type (0=Dry, 1=Fresh)')
    ax2.set_ylabel('Time on Shelf (days)')
    ax2.set_zlabel('Food Quality (%)')
    fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)

    # Save the combined plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    return base64.b64encode(img.getvalue()).decode('utf8')


def classify_quality(quality_value):
    if quality_value > 90:
        return "Excellent"
    elif quality_value > 70:
        return "Good"
    elif quality_value > 30:
        return "Fair"
    else:
        return "Poor"

@app.route("/", methods=["GET", "POST"])
def index():
    error_message = None

    if request.method == "POST":
        try:
            # Get the input values from the form
            temperature_value = float(request.form["temperature"])
            humidity_value = float(request.form["humidity"])
            food_type_value = int(request.form["food_type"])
            time_on_shelf_value = float(request.form["time_on_shelf"])

            # Validate input
            if not (0 <= temperature_value <= 30):
                error_message = "Please enter a temperature between 0 and 30 °C."
            elif not (0 <= humidity_value <= 100):
                error_message = "Please enter a humidity level between 0 and 100%."
            elif not (0 <= food_type_value <= 1):
                error_message = "Please enter 0 for Dry/Canned food or 1 for Fresh food."
            elif not (0 <= time_on_shelf_value <= 30):
                error_message = "Please enter the time on shelf between 0 and 30 days."
            else:
                # Compute the food quality if inputs are valid
                quality_value = compute_food_quality(temperature_value, humidity_value, food_type_value, time_on_shelf_value)
                quality_label = classify_quality(quality_value)

                # Generate the main output plot
                temperature, humidity, food_type, time_on_shelf, quality = initialize_system()
                plot_img = plot_membership_functions(temperature, humidity, food_type, time_on_shelf, quality, quality_value)

                # Generate individual membership function plots
                individual_plots = plot_individual_membership_functions(temperature, humidity, food_type, time_on_shelf)

                # Generate the 3D surface plot
                surface_plot_img = plot_3d_surface(temperature, humidity, food_type, time_on_shelf)

                return render_template("index.html", quality=quality_value, quality_label=quality_label,
                                       plot_img=plot_img, individual_plots=individual_plots,
                                       surface_plot_img=surface_plot_img)

        except ValueError:
            error_message = "Please enter valid numeric values for all fields."

    # Render the page with an error message if present
    return render_template("index.html", error_message=error_message)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)

