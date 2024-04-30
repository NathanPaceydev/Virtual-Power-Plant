import subprocess
import sys

def install_packages():
    try:
        # List of packages to install
        packages = [
            "Flask",
            "plotly",
            "openmeteo-requests",
            "matplotlib",
            "requests-cache",
            "retry-requests",
            "numpy",
            "pandas",
            "scipy"
        ]
        
        # Installing packages
        for package in packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        
        # Upgrading pandas
        #subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pandas"])
        
        # Running the application script
        subprocess.check_call([sys.executable, "app.py"])
    
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing packages or running the script: {e}")

if __name__ == "__main__":
    install_packages()
