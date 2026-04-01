# Step 1: Use a lightweight Python version
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR /app

# Step 3: Copy only the requirements first (this is a "Future Tip" for faster builds)
COPY requirements.txt .

# Step 4: Install the libraries
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all your project files into the container
COPY . .

# Step 6: Train the model during the image build so the .pkl is "cached"
RUN python train_model.py

# Step 7: Expose the port Flask uses
EXPOSE 5000

# Step 8: Start the app using Gunicorn (Standard for hosting)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]