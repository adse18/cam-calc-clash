from app import create_app
import warnings
warnings.filterwarnings("ignore", module="torch")

app = create_app()

if __name__ == "__main__":
    app.run(debug=True)
