FROM python:3.9

# Create a non-root user
RUN useradd -m -u 1000 user

# Switch to the new user
USER user

# Set environment variables
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory
WORKDIR $HOME/app

# Copy application files with the correct ownership
COPY --chown=user . $HOME/app

# Copy requirements.txt and install dependencies
COPY --chown=user ./requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the data directory exists and copy the file
RUN mkdir -p $HOME/app/data
COPY --chown=user ./data/airbnb_file.pdf $HOME/app/data/airbnb_file.pdf

# Set the command to run the application
CMD ["chainlit", "run", "app.py", "--port", "7860"]
