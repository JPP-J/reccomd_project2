# Step 1: Use an official Python runtime as a parent image
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy the current directory contents into the container
COPY . .

# Step 4:Update pip to the latest version
RUN pip install --upgrade pip

# Step 5: Install any needed dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 6: Specify the command to run the main script
CMD ["python", "app.py"]
