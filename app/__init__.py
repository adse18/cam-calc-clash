import os
from flask import Flask, render_template

def create_app(test_config=None):
    # Create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    
    # Basic configuration
    app.config.from_mapping(
        SECRET_KEY='dev',
    )

    # Load the instance config if it exists
    if test_config is None:
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    # Ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Route for the index page
    @app.route('/')
    def index():
        return render_template('index.html')
    
    # Import and register additional routes
    with app.app_context():
        from .routes import bp as main_bp
        app.register_blueprint(main_bp)
    
    return app
