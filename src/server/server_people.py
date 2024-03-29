from flask import render_template
import connexion

# Creat the application instance
app = connexion.App(__name__, specification_dir = './')

# Read the swagger.yml file to configure the endpoints
app.add_api('swagger_people.yml')

# Create a URL route in our application for "/"
@app.route('/')
def home():
    """
    The function just responds to the browser URL
    localhost:5000/

    """
    return render_template('home.html')

# If we're running in the stand-alone mode, run the application
if __name__ == '__main__':
    app.run(host='0.0.0.0', port = 5000, debug=True)