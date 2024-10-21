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