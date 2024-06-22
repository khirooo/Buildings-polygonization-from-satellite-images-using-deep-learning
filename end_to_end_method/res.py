import shapefile
import matplotlib.pyplot as plt

sf = shapefile.Reader("/home/sagemaker-user/Caerus/Yanis/000000000047.dbf")
fig, ax = plt.subplots()

# Loop through each shape in the shapefile
for shape in sf.shapes():
    # Get the points of the shape
    points = shape.points
    parts = shape.parts

    # Loop through each part of the shape
    for i in range(len(parts)):
        start = parts[i]
        if i < len(parts) - 1:
            end = parts[i + 1]
        else:
            end = len(points)
        
        # Extract x and y coordinates for this part
        x = [point[0] for point in points[start:end]]
        y = [point[1] for point in points[start:end]]

        # Plot the shape part
        ax.plot(x, y, 'k')  # 'k' stands for black colo

# Set plot title and labels
ax.set_title('Shapefile Plot')
ax.set_xlabel('x')
ax.set_ylabel('y')

# Display the plot
plt.savefig('polygon2.png')